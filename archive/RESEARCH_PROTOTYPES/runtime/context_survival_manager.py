from typing import Dict, Any

class ContextSurvivalManager:
    """
    Ensures long-session context integrity and retrieval continuity.
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.consecutive_failures = 0

    def verify_integrity(self, step_results: Dict[str, Any]):
        """
        Checks if the retrieval score has dropped below critical levels.
        """
        score = step_results.get("retrieval_score", 1.0)
        
        if score < self.threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures > 5:
                print(f"WARNING: Context integrity compromised! Score: {score:.4f}")
        else:
            self.consecutive_failures = 0
            
        return self.consecutive_failures <= 5
