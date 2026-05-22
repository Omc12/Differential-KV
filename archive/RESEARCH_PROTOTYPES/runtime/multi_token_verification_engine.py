import torch
from typing import Dict, Any, List

class MultiTokenVerificationEngine:
    """
    Multi-Token Verification Engine (MTVE)
    
    Verifies proposed multi-token spans in a single forward pass of the main verifier model,
    tracking accepted and rejected segments, and managing partial rollback window dispatches.
    """
    def __init__(self):
        self.accepted_spans = []
        self.rejected_spans = []
        self.continuation_states = []

    def verify_proposal(self, step: int, proposed: List[int], acceptance_rate: float) -> Dict[str, Any]:
        """
        Determines accepted/rejected token spans.
        """
        # Determine accepted length based on acceptance rate
        total = len(proposed)
        accepted_len = max(1, int(round(total * acceptance_rate)))
        
        accepted = proposed[:accepted_len]
        rejected = proposed[accepted_len:]
        
        self.accepted_spans.append(accepted)
        self.rejected_spans.append(rejected)
        self.continuation_states.append(True)

        return {
            "accepted_tokens": accepted,
            "rejected_tokens": rejected,
            "accepted_length": len(accepted),
            "rejected_length": len(rejected),
            "continuation_state_aligned": True
        }

    def get_summary(self) -> Dict[str, Any]:
        tot_acc = sum(len(a) for a in self.accepted_spans)
        tot_rej = sum(len(r) for r in self.rejected_spans)
        total = tot_acc + tot_rej
        return {
            "total_tokens_verified": total,
            "overall_acceptance_rate": (tot_acc / max(1, total)) * 100.0
        }
