import time
from typing import Dict, Any
from .metric_range_assertions import validate_retrieval_score, validate_retention_percent

class RetrievalMetricSanity:
    """
    Eliminates benchmark inflation and hidden accounting errors in retrieval.
    """
    
    def __init__(self):
        self.start_time = None
        self.orchestration_overhead = 0.0

    def start_timing(self):
        self.start_time = time.perf_counter()

    def record_orchestration_step(self, duration: float):
        self.orchestration_overhead += duration

    def audit_retrieval_event(self, 
                              retrieval_score: float, 
                              retention_percent: float, 
                              is_sparse: bool,
                              raw_latency: float,
                              migration_latency: float = 0.0):
        """
        Comprehensive audit of a single retrieval event.
        """
        # 1. Hard Range Bounds
        validate_retrieval_score(retrieval_score)
        validate_retention_percent(retention_percent)
        
        # 2. Sparse Collapse Detection (Honesty Check)
        if is_sparse and retention_percent >= 100.0:
            raise ValueError("CRITICAL ERROR: Sparse mode reported 100% retention. This is hidden fallback-to-dense logic.")
            
        if not is_sparse and retention_percent < 100.0:
            raise ValueError("CRITICAL ERROR: Dense mode reported <100% retention. Accounting error.")

        # 3. TPS Accounting Completeness
        # Total latency must include migration and orchestration
        total_latency = raw_latency + migration_latency + self.orchestration_overhead
        
        # Reset overhead for next event if needed, or keep cumulative
        # For now, we just ensure it's positive
        if total_latency <= 0:
            raise ValueError("CRITICAL ERROR: Non-positive total latency detected. Clock error.")

        return {
            "status": "VALID",
            "total_latency": total_latency,
            "overhead_ratio": (migration_latency + self.orchestration_overhead) / total_latency if total_latency > 0 else 0
        }

    @staticmethod
    def detect_benchmark_contamination(retrieval_score: float, expected_max: float = 0.999):
        """
        Detects suspiciously high retrieval scores that might indicate test set leakage
        or silent cache reuse.
        """
        if retrieval_score > expected_max:
            print(f"WARNING: Retrieval score {retrieval_score} is extremely high (> {expected_max}). "
                  "Check for benchmark contamination or cache leakage.")

if __name__ == "__main__":
    print("Running RetrievalMetricSanity self-test...")
    auditor = RetrievalMetricSanity()
    
    # Valid sparse event
    res = auditor.audit_retrieval_event(0.92, 15.0, is_sparse=True, raw_latency=0.01)
    print(f"[PASS] Valid sparse event: {res}")
    
    # Test: Sparse reported as 100% (Inflation)
    try:
        auditor.audit_retrieval_event(0.99, 100.0, is_sparse=True, raw_latency=0.01)
    except ValueError as e:
        print(f"[PASS] Caught sparse-to-dense inflation: {e}")
        
    # Test: Out of bounds
    try:
        auditor.audit_retrieval_event(1.5, 50.0, is_sparse=True, raw_latency=0.01)
    except ValueError as e:
        print(f"[PASS] Caught out of bounds score: {e}")
        
    print("RetrievalMetricSanity validated.")
