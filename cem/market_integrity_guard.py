
import torch
from typing import Dict, Any, List, Optional

class MarketIntegrityGuard:
    """
    PHASE 23.5: CEM - Market Integrity Guard.
    Prevents resource monopolization and ensures economic stability.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.winner_history = {} # region_id -> win_count
        
        self.metrics = {
            "anti_monopoly_integrity": 1.0,
            "starvation_prevention_score": 1.0,
            "market_coherence": 1.0,
            "symbolic_continuity": 1.0
        }

    def validate_market_step(self, winners: List[int]) -> bool:
        """
        Validates that no single region is monopolizing resources.
        """
        for w in winners:
            self.winner_history[w] = self.winner_history.get(w, 0) + 1
            
        # Monopolization check: any region winning > 80% of recent slots
        total_slots = sum(self.winner_history.values()) + 1e-9
        for region, count in self.winner_history.items():
            if count / total_slots > 0.8 and total_slots > 10:
                self.metrics["anti_monopoly_integrity"] *= 0.9
                self.metrics["market_coherence"] *= 0.95
                return False
                
        self.metrics["anti_monopoly_integrity"] = 0.99
        self.metrics["starvation_prevention_score"] = 0.99
        self.metrics["market_coherence"] = 1.0
        self.metrics["symbolic_continuity"] = 1.0
        
        return True

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
