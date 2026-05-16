
import torch
from typing import Dict, List, Any

class ServingIntegrityGuard:
    """
    PHASE 24.1: Serving Integrity Guard (BSO).
    Ensures cross-request isolation and leakage prevention during concurrent serving.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.isolation_violations = 0
        self.checked_requests = 0
        
    def validate_isolation(self, 
                           request_id: str, 
                           executed_tokens: torch.Tensor, 
                           allowed_symbolic_domain: torch.Tensor) -> bool:
        """
        Verifies that a request's execution stayed within its allowed symbolic domain.
        """
        self.checked_requests += 1
        
        # 1. Domain leakage check
        # Ensure executed tokens are subset of allowed symbolic domain (simplified)
        # In prod, this would use cryptographic or hardware-enforced boundaries
        leakage = torch.any((executed_tokens > 0) & (allowed_symbolic_domain == 0))
        
        if leakage:
            self.isolation_violations += 1
            return False
            
        return True

    def get_integrity_metrics(self) -> Dict[str, Any]:
        isolation_score = 1.0 - (self.isolation_violations / self.checked_requests) if self.checked_requests > 0 else 1.0
        return {
            "cross_request_isolation": isolation_score,
            "leakage_events": self.isolation_violations,
            "validation_count": self.checked_requests
        }
