from typing import Dict, List

class AnchorDecayAuditor:
    """
    Monitors the survival and attention weight of critical anchors over time.
    """
    def __init__(self):
        self.anchor_history = {} # anchor_id -> [weights]

    def audit_anchors(self, current_anchors: Dict[str, float]):
        for aid, weight in current_anchors.items():
            if aid not in self.anchor_history:
                self.anchor_history[aid] = []
            self.anchor_history[aid].append(weight)

    def get_decay_rates(self) -> Dict[str, float]:
        decay_rates = {}
        for aid, weights in self.anchor_history.items():
            if len(weights) < 2:
                continue
            # Simple linear decay estimate
            decay = (weights[-1] - weights[0]) / len(weights)
            decay_rates[aid] = decay
        return decay_rates
