class PlaceholderOutputGuard:
    """
    Prevents placeholder outputs from entering the benchmark reporting stack.
    Raises an error if synthetic generation is detected.
    """
    def __init__(self, detector):
        self.detector = detector

    def guard_result(self, result: dict):
        text = result.get("text", "")
        tokens = result.get("token_ids", [])
        
        if self.detector.is_synthetic(text):
            raise ValueError(f"CRITICAL ERROR: Synthetic text detected: '{text}'")
            
        if not self.detector.validate_stream(tokens):
            raise ValueError(f"CRITICAL ERROR: Synthetic token stream detected: {tokens[:10]}...")
        
        return True
