import time
import torch

class EndToEndTruthMeter:
    """
    PHASE 6H: End-to-End Truth Meter
    The definitive judge of Phase 6 success.
    Measures E2E latency from token input to output logit, 
    verifying that no overhead is hidden.
    """
    def __init__(self):
        self.results = []

    def verify_run(self, model_run_func, input_tokens):
        """
        Runs a full inference step and measures TOTAL wall time.
        """
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        logits = model_run_func(input_tokens)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        total_time = end - start
        return total_time, logits
