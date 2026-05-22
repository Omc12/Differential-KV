import torch

class ContextIntegrityChecker:
    """
    Verifies that the context window is correctly utilized and not truncated.
    MANDATORY for long-context (16k, 32k) validation.
    """
    def __init__(self, max_context: int = 32768):
        self.max_context = max_context

    def check_sequence_length(self, input_ids):
        length = input_ids.shape[1]
        if length > self.max_context:
            return False, f"Sequence length {length} exceeds max context {self.max_context}"
        return True, length

    def verify_kv_cache_integrity(self, pkv, expected_layers: int):
        """
        Ensures the KV cache contains the correct number of layers.
        """
        if pkv is None:
            return False, "KV cache is empty"
            
        if len(pkv) != expected_layers:
            return False, f"KV cache layer mismatch: expected {expected_layers}, got {len(pkv)}"
            
        return True, len(pkv)
