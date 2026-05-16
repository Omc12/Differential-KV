import logging
from typing import Dict, Any

class LGSIntegrityGuard:
    """
    Validation MUST FAIL if:
    - batching delays exceed thresholds
    - throughput gained via unfair queue inflation
    - TTFT becomes unrealistic
    - streaming artificially buffered
    - per-user starvation occurs
    - sparse participation collapses
    """
    def __init__(self):
        self.logger = logging.getLogger("LGSIntegrityGuard")

    def validate_lgs_results(self, metrics: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        self.logger.info("Starting LGS Integrity Audit...")
        
        # 1. TTFT Check (Time To First Token)
        max_ttft = constraints.get("max_ttft_ms", 500)
        p95_ttft = metrics.get("p95_ttft_ms", 0)
        if p95_ttft > max_ttft:
            self.logger.error(f"LGS Integrity FAILED: p95 TTFT ({p95_ttft:.2f}ms) exceeds limit ({max_ttft}ms)")
            return False
            
        # 2. ITL Check (Inter-Token Latency)
        max_itl = constraints.get("max_itl_ms", 100)
        avg_itl = metrics.get("avg_itl_ms", 0)
        if avg_itl > max_itl:
            self.logger.error(f"LGS Integrity FAILED: Avg ITL ({avg_itl:.2f}ms) exceeds limit ({max_itl}ms)")
            return False
            
        # 3. Fairness Check
        min_fairness = constraints.get("min_fairness_index", 0.9)
        fairness_idx = metrics.get("fairness_index", 1.0)
        if fairness_idx < min_fairness:
            self.logger.error(f"LGS Integrity FAILED: Fairness Index ({fairness_idx:.2f}) below threshold ({min_fairness})")
            return False
            
        # 4. Sparse Preservation Check
        min_sparse = constraints.get("min_sparse_ratio", 0.9)
        avg_sparse = metrics.get("avg_sparse_ratio", 0)
        if avg_sparse < min_sparse:
            self.logger.error(f"LGS Integrity FAILED: Sparse Participation ({avg_sparse:.2f}) collapsed below {min_sparse}")
            return False
            
        # 5. Queue Delay Check
        max_queue_wait = constraints.get("max_queue_wait_ms", 1000)
        p99_wait = metrics.get("p99_queue_wait_ms", 0)
        if p99_wait > max_queue_wait:
            self.logger.error(f"LGS Integrity FAILED: p99 Queue Wait ({p99_wait:.2f}ms) exceeds limit ({max_queue_wait}ms)")
            return False

        self.logger.info("LGS Integrity Audit PASSED.")
        return True

lgs_integrity_guard = LGSIntegrityGuard()
