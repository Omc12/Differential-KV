class LongHorizonContaminationGuard:
    """
    Detects hidden state leakage between sessions during long runs.
    """
    def __init__(self):
        self.seen_signatures = set()

    def check_leakage(self, current_signature: str) -> bool:
        if current_signature in self.seen_signatures:
            print(f"CRITICAL: Hidden state contamination detected! Signature {current_signature} seen before.")
            return True
        self.seen_signatures.add(current_signature)
        return False
