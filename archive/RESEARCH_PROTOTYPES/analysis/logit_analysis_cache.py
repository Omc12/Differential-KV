
import torch
import numpy as np
from typing import Dict, Any, Optional

class LogitAnalysisCache:
    """
    PPOSAH Phase 20.6A: Shared Logit Analysis Cache.
    Prevents repeated 152k softmax and entropy calculations.
    """
    def __init__(self):
        self.logits: Optional[torch.Tensor] = None
        self.probs: Optional[torch.Tensor] = None
        self.entropy: Optional[float] = None
        self.top_k_ids: Optional[torch.Tensor] = None
        self.top_k_probs: Optional[torch.Tensor] = None
        
    def update(self, logits: torch.Tensor):
        self.logits = logits
        self.probs = torch.softmax(logits.float().squeeze(0), dim=-1)
        
        # Entropy
        log_probs = torch.log(self.probs + 1e-12)
        self.entropy = -(self.probs * log_probs).sum().item()
        
        # Top-k
        self.top_k_probs, self.top_k_ids = torch.topk(self.probs, k=5)

    def reset(self):
        self.logits = None
        self.probs = None
        self.entropy = None
        self.top_k_ids = None
        self.top_k_probs = None

class FusedPrecisionTelemetry:
    """
    PPOSAH Phase 20.6A: Fused Precision Telemetry.
    Eliminates GPU stalls by merging metrics into a single synchronization point.
    """
    def __init__(self):
        self.telemetry_history = []

    def measure(self, cache: LogitAnalysisCache, expected_id: int, stab_factor: float) -> Dict[str, Any]:
        """
        Calculates all symbolic precision metrics using the cached distribution.
        Only one .item() sync (hidden in cache.entropy) plus any additional derived logic.
        """
        if cache.probs is None:
            return {}

        expected_prob = cache.probs[expected_id].item()
        top_id = cache.top_k_ids[0].item()
        
        # Drift Risk logic
        risk = 0.0
        if expected_prob < 0.5: risk += 0.4
        if expected_prob < 0.1: risk += 0.4
        if top_id != expected_id: risk += 0.2
        
        metrics = {
            "entropy_nats": cache.entropy,
            "drift_risk": risk,
            "expected_prob": expected_prob,
            "stab_factor": stab_factor,
            "is_collapsed": cache.entropy < 0.05,
            "top_match": top_id == expected_id
        }
        
        self.telemetry_history.append(metrics)
        return metrics

    def reset(self):
        self.telemetry_history = []
