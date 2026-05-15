
from typing import Set, List

class DelimiterIntegrityGuard:
    """
    PHASE 21.3: STRL - Delimiter Integrity Guard.
    Tracks structural boundary health and stabilizes framing.
    """
    def __init__(self, delimiter_ids: Set[int]):
        self.delimiter_ids = delimiter_ids
        self.integrity_score = 1.0
        self.violation_count = 0

    def record_token(self, token_id: int, is_expected_structural_token: bool):
        """Updates integrity based on whether the token matches structural expectations."""
        is_delimiter = token_id in self.delimiter_ids
        
        if is_delimiter:
            if is_expected_structural_token:
                # Strong structural confirmation
                self.integrity_score = min(1.0, self.integrity_score + 0.1)
            else:
                # Delimiter in a non-structural position (noise)
                self.integrity_score = max(0.0, self.integrity_score - 0.2)
                self.violation_count += 1
        elif is_expected_structural_token:
            # Missing delimiter at a structural boundary
            self.integrity_score = max(0.0, self.integrity_score - 0.3)
            self.violation_count += 1

    def get_stabilization_bias(self) -> float:
        """Returns a bias strength for structural reinforcement."""
        if self.integrity_score < 0.6:
            return 2.0 * (1.0 - self.integrity_score)
        return 0.0
