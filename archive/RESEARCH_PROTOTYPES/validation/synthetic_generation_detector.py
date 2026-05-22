import re

class SyntheticGenerationDetector:
    """
    Detects if the generated output is synthetic (e.g., "token_0", "token_1").
    Enforces real model output.
    """
    def __init__(self):
        self.synthetic_patterns = [
            r"token_\d+",
            r"placeholder_token",
            r"mock_token",
            r"^token$"
        ]

    def is_synthetic(self, text: str) -> bool:
        for pattern in self.synthetic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def validate_stream(self, tokens: list) -> bool:
        """
        Check if tokens look like real model output (e.g., not just incrementing IDs).
        """
        if not tokens:
            return False
        # Very simple check: if more than 90% are sequential integers, it's likely synthetic
        sequential = 0
        for i in range(len(tokens) - 1):
            if tokens[i+1] == tokens[i] + 1:
                sequential += 1
        return (sequential / len(tokens)) < 0.9
