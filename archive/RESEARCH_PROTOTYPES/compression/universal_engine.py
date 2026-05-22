import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple
import sys
from pathlib import Path
from transformers.cache_utils import DynamicCache

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compression.shared_basis import SharedBasisManager
from compression.adaptive import AdaptiveRankSelector
from compression.quantization import quantize_int8, dequantize_int8
from anchor_logic.semantic_anchor_system import SemanticAnchorMemory, SemanticReinjector, PositionAwarePolicy
from anchor_logic.active_repair_controller import ActiveRepairController

class UniversalCompressionEngine:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sb_manager = SharedBasisManager()
        self.rank_selector = AdaptiveRankSelector(rank_buckets=[8, 16, 32, 64], method="energy")
        
        # SAM setup
        self.sam_memory = SemanticAnchorMemory(max_anchors=256)
        self.sam_reinjector = SemanticReinjector(self.sam_memory)
        self.sam_policy = PositionAwarePolicy(interval=64)
        
        # ACTR setup
        self.arc = ActiveRepairController(self.sam_memory)
        
    def compress_kv(self, past_kv: Any, mode: str) -> Tuple[Any, Dict[str, Any]]:
        mode = mode.lower()
        is_cache_obj = isinstance(past_kv, DynamicCache)
        
        if mode == "fp16":
            return past_kv, {"ratio": 1.0}
            
        if mode == "int8":
            recon_tuple, stats = self._compress_diff_int8(past_kv)
        elif mode == "rank8":
            recon_tuple, stats = self._compress_shared_basis(past_kv, rank=8)
        elif mode == "sam":
            recon_tuple, stats = self._apply_sam(past_kv)
        elif mode == "actr":
            recon_tuple, stats = self._apply_actr(past_kv)
        elif mode == "lcg":
            recon_tuple, stats = self._compress_lcg(past_kv)
        else:
            return past_kv, {"ratio": 1.0}

        if is_cache_obj:
            new_cache = DynamicCache()
            for i, (k, v) in enumerate(recon_tuple):
                new_cache.update(k, v, layer_idx=i)
            return new_cache, stats
        else:
            return recon_tuple, stats

    def _apply_sam(self, past_kv):
        return self._compress_shared_basis(past_kv, rank=8)

    def _apply_actr(self, past_kv):
        return self._compress_shared_basis(past_kv, rank=8)

    def _compress_diff_int8(self, past_kv):
        recon_kv = []
        total_orig = 0
        total_comp = 0
        kv_list = list(past_kv) if isinstance(past_kv, DynamicCache) else past_kv
        for layer in kv_list:
            k, v = layer[0], layer[1]
            total_orig += k.numel() * 2 + v.numel() * 2
            def q_dq(t):
                scale = t.abs().max() / 127.0
                q = (t / (scale + 1e-6)).round().clamp(-128, 127).to(torch.int8)
                dq = q.to(torch.float16) * scale
                return dq, q.numel() + 4
            rk, sk = q_dq(k)
            rv, sv = q_dq(v)
            recon_kv.append((rk, rv))
            total_comp += sk + sv
        return tuple(recon_kv), {"ratio": total_orig / total_comp}

    def _compress_shared_basis(self, past_kv, rank=8):
        recon_kv = []
        total_orig = 0
        total_comp = 0
        kv_list = list(past_kv) if isinstance(past_kv, DynamicCache) else past_kv
        for i, layer in enumerate(kv_list):
            k, v = layer[0], layer[1]
            total_orig += k.numel() * 2 + v.numel() * 2
            b, h, s, d = k.shape
            k_flat = k.transpose(1, 2).reshape(b * s, h * d).float()
            v_flat = v.transpose(1, 2).reshape(b * s, h * d).float()
            basis_k = self.sb_manager.create_basis(k_flat, rank, f"L{i}_K")
            basis_v = self.sb_manager.create_basis(v_flat, rank, f"L{i}_V")
            ck = self.sb_manager.compress_block(k_flat, f"L{i}_K", rank=rank)
            cv = self.sb_manager.compress_block(v_flat, f"L{i}_V", rank=rank)
            rk = self.sb_manager.decompress_block(ck).to(self.device).to(torch.float16)
            rv = self.sb_manager.decompress_block(cv).to(self.device).to(torch.float16)
            rk = rk.reshape(b, s, h, d).transpose(1, 2)
            rv = rv.reshape(b, s, h, d).transpose(1, 2)
            recon_kv.append((rk, rv))
            total_comp += ck.nbytes() + cv.nbytes()
        return tuple(recon_kv), {"ratio": total_orig / total_comp if total_comp > 0 else 1.0}

    def _compress_lcg(self, past_kv):
        recon_kv = []
        total_orig = 0
        total_comp = 0
        kv_list = list(past_kv) if isinstance(past_kv, DynamicCache) else past_kv
        for i, layer in enumerate(kv_list):
            k, v = layer[0], layer[1]
            total_orig += k.numel() * 2 + v.numel() * 2
            b, h, s, d = k.shape
            k_flat = k.transpose(1, 2).reshape(b * s, h * d).float()
            v_flat = v.transpose(1, 2).reshape(b * s, h * d).float()
            basis_k = self.sb_manager.create_basis(k_flat, 32, f"L{i}_K_LCG")
            basis_v = self.sb_manager.create_basis(v_flat, 32, f"L{i}_V_LCG")
            rk_val = self.rank_selector.select_rank(k_flat)
            rv_val = self.rank_selector.select_rank(v_flat)
            ck = self.sb_manager.compress_block(k_flat, f"L{i}_K_LCG", rank=rk_val, sparse_ratio=0.05)
            cv = self.sb_manager.compress_block(v_flat, f"L{i}_V_LCG", rank=rv_val, sparse_ratio=0.05)
            rk = self.sb_manager.decompress_block(ck).to(self.device).to(torch.float16)
            rv = self.sb_manager.decompress_block(cv).to(self.device).to(torch.float16)
            rk = rk.reshape(b, s, h, d).transpose(1, 2)
            rv = rv.reshape(b, s, h, d).transpose(1, 2)
            recon_kv.append((rk, rv))
            total_comp += ck.nbytes() + cv.nbytes()
        return tuple(recon_kv), {"ratio": total_orig / total_comp if total_comp > 0 else 1.0}
