import os
import torch
import threading
from typing import Optional, List, Dict, Any

_BYPASS = os.environ.get('DIFFKV_MLA_LATENT', '0') != '1'

class MLAProjector:
    def __init__(
        self,
        head_dim: int,
        kv_heads: int,
        latent_dim: int = 0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float16,
        n_calib_blocks: int = 16
    ):
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.feat_dim = 2 * kv_heads * head_dim
        
        if latent_dim == 0:
            self.latent_dim = max(8, (kv_heads * head_dim) // 4)
        else:
            self.latent_dim = latent_dim
            
        self.device = device
        self.dtype = dtype
        self.n_calib_blocks = n_calib_blocks
        
        self.W: Optional[torch.Tensor] = None
        self._calib_data: List[torch.Tensor] = []
        self._calib_count: int = 0
        self.is_calibrated: bool = False
        self._lock = threading.Lock()

    def update_calibration(self, deltas: torch.Tensor) -> bool:
        if _BYPASS:
            return True
            
        with self._lock:
            if self.is_calibrated:
                return True
                
            self._calib_data.append(deltas.detach().cpu())
            self._calib_count += 1
            
            if self._calib_count >= self.n_calib_blocks:
                self._init_from_pca()
                self.is_calibrated = True
                return True
                
        return False

    def _init_from_pca(self):
        try:
            stacked = torch.cat(self._calib_data, dim=0).to(torch.float32)
            # SVD: full_matrices=False
            U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
            
            # top-latent_dim right singular vectors
            # Vh shape: [min(N, feat_dim), feat_dim]
            W = Vh[:self.latent_dim, :].T  # shape: [feat_dim, latent_dim]
            self.W = W.to(device=self.device, dtype=self.dtype)
        except Exception as e:
            print(f'[DiffKV MLA] PCA init failed: {e}')
            self.W = None
        finally:
            self._calib_data = []

    def project(self, deltas: torch.Tensor) -> torch.Tensor:
        if _BYPASS or self.W is None:
            return deltas
        with self._lock:
            if self.W is None:
                return deltas
            w = self.W.to(deltas.device, deltas.dtype)
            return deltas @ w

    def unproject(self, latent: torch.Tensor) -> torch.Tensor:
        if _BYPASS or self.W is None:
            return latent
        with self._lock:
            if self.W is None:
                return latent
            w = self.W.to(latent.device, latent.dtype)
            return latent @ w.T

    def to(self, device_or_dtype):
        with self._lock:
            if isinstance(device_or_dtype, torch.dtype):
                self.dtype = device_or_dtype
                if self.W is not None:
                    self.W = self.W.to(dtype=self.dtype)
            elif isinstance(device_or_dtype, str) or isinstance(device_or_dtype, torch.device):
                self.device = str(device_or_dtype)
                if self.W is not None:
                    self.W = self.W.to(device=self.device)
            return self

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'W': self.W.clone() if self.W is not None else None,
                'latent_dim': self.latent_dim,
                'feat_dim': self.feat_dim,
                'is_calibrated': self.is_calibrated
            }

    def load_state_dict(self, d: Dict[str, Any]):
        with self._lock:
            self.W = d.get('W')
            if self.W is not None:
                self.W = self.W.to(device=self.device, dtype=self.dtype)
            self.latent_dim = d.get('latent_dim', self.latent_dim)
            self.feat_dim = d.get('feat_dim', self.feat_dim)
            self.is_calibrated = d.get('is_calibrated', self.is_calibrated)
