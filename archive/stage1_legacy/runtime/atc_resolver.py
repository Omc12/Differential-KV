import torch
from typing import Dict, Any, Optional

from adaptive_token_survival_controller import AdaptiveTokenSurvivalController
from active_sequence_compressor import ActiveSequenceCompressor
from sparse_token_router import SparseTokenRouter
from triton_token_collapse_kernel import atc_kernel
from token_survival_telemetry import atc_telemetry
from atc_integrity_guard import guard
from sparse_flop_accountant import accountant
from runtime_density_profiler import profiler

class ATCResolver:
    """
    Main resolver for ATC (Adaptive Token Collapse).
    Extends and strengthens existing sparse systems (SEM, SML, SHM).
    """
    def __init__(self, target_ratio: float = 0.5):
        self.controller = AdaptiveTokenSurvivalController(target_ratio=target_ratio)
        self.compressor = ActiveSequenceCompressor()
        self.router = SparseTokenRouter()
        print(f"[ATC] Resolver initialized (Target Survival: {target_ratio:.2f})")

    def resolve_token_survival(
        self, 
        x: torch.Tensor, 
        keys: torch.Tensor, 
        queries: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes token survival and sequence collapse in the production path.
        """
        profiler.start("attention") # Reusing profiler scope
        
        # 1. Score and Filter
        scores = self.controller.score_tokens(keys, queries)
        mask = self.controller.filter_active_tokens(scores)
        
        # 2. Sequence Collapse (Hardware-visible via Triton)
        bsz, seq_len, d = x.shape
        _, indices = self.compressor.compress_sequence(x, mask)
        compressed_x = atc_kernel.gather_active_tokens(x, indices)
        
        # 3. Telemetry and Accounting
        atc_telemetry.record_step(seq_len, indices.numel())
        accountant.record_attention(1, seq_len, d, 1, indices.numel()) # Approximate
        
        # 4. Routing to compute
        # In ATC, only compressed_x enters further layers
        
        profiler.end("attention")
        return compressed_x

    def get_atc_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive ATC report.
        """
        metrics = atc_telemetry.get_metrics()
        telemetry = atc_kernel.get_telemetry()
        
        report = {
            **metrics,
            **telemetry
        }
        
        guard.validate_atc_state(report)
        guard.check_integrity()
        
        return report
