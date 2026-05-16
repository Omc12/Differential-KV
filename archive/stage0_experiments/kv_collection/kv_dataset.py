"""
kv_collection/kv_dataset.py

KVDataset: persistent storage and loading utilities for KV snapshots.

Provides:
  - Load .pt KV files and their metadata
  - Filter by model_name, text_type, seq_len
  - Iterate over layers across multiple snapshots
  - Compare real vs synthetic KV statistics
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch


class KVDataset:
    """
    Loads and queries a collection of saved KV snapshots.

    Parameters
    ----------
    root_dir : str — directory containing _kv.pt and _meta.json files
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self._index: List[Dict] = []
        self._load_index()

    def _load_index(self):
        """Scan root_dir for all snapshot metadata files."""
        self._index = []
        for meta_path in sorted(self.root.glob("**/*_meta.json")):
            kv_path = Path(str(meta_path).replace("_meta.json", "_kv.pt"))
            if kv_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["_kv_path"]   = str(kv_path)
                meta["_meta_path"] = str(meta_path)
                self._index.append(meta)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        models = set(m.get("model_name", "?") for m in self._index)
        types  = set(m.get("text_type",  "?") for m in self._index)
        return (f"KVDataset(root={self.root}, n={len(self._index)}, "
                f"models={models}, types={types})")

    def filter(
        self,
        model_name: Optional[str] = None,
        text_type:  Optional[str] = None,
        min_seq_len: int = 0,
        max_seq_len: int = 999999,
    ) -> "KVDataset":
        """Return a filtered view (new KVDataset-like object)."""
        filtered = KVDataset.__new__(KVDataset)
        filtered.root   = self.root
        filtered._index = [
            m for m in self._index
            if (model_name is None or m.get("model_name") == model_name)
            and (text_type  is None or m.get("text_type")  == text_type)
            and min_seq_len <= m.get("seq_len", 0) <= max_seq_len
        ]
        return filtered

    def iter_snapshots(self) -> Iterator[Tuple[Dict, Dict[int, torch.Tensor]]]:
        """Yield (metadata, kv_by_layer) for each snapshot."""
        for meta in self._index:
            kv_path = meta["_kv_path"]
            kv_by_layer: Dict[int, torch.Tensor] = torch.load(kv_path,
                                                               map_location="cpu",
                                                               weights_only=True)
            yield meta, kv_by_layer

    def iter_layers(self, layer_idx: int) -> Iterator[Tuple[Dict, torch.Tensor]]:
        """Yield (metadata, kv_tensor) for a specific layer across all snapshots."""
        for meta, kv_by_layer in self.iter_snapshots():
            if layer_idx in kv_by_layer:
                yield meta, kv_by_layer[layer_idx]

    def summary(self) -> List[Dict]:
        return [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in self._index
        ]

    def stats_comparison(
        self, synthetic_kv: torch.Tensor, layer_idx: int = 0
    ) -> Dict:
        """
        Compare statistics of real KV (from dataset) vs a synthetic KV tensor.

        Returns dict with side-by-side statistics.
        """
        import math, numpy as np

        def _rms(t: torch.Tensor) -> float:
            return (t.float().norm() / math.sqrt(t.numel())).item()

        def _stats(kv: torch.Tensor) -> Dict:
            f = kv.float()
            flat = f.flatten()
            consec_rms = []
            for i in range(1, min(kv.shape[0], 200)):
                diff = kv[i].float() - kv[i-1].float()
                consec_rms.append(_rms(diff))
            return {
                "mean":          round(float(flat.mean()), 5),
                "std":           round(float(flat.std()),  5),
                "rms":           round(_rms(f), 5),
                "p95_abs":       round(float(torch.quantile(flat.abs(), 0.95)), 5),
                "smoothness":    round(float(np.mean(consec_rms)) if consec_rms else 0, 5),
            }

        result = {"synthetic": _stats(synthetic_kv)}
        real_stats = []
        for meta, kv in self.iter_layers(layer_idx):
            rs = _stats(kv)
            rs["model"] = meta.get("model_name", "?")
            rs["type"]  = meta.get("text_type",  "?")
            real_stats.append(rs)

        result["real"] = real_stats
        return result
