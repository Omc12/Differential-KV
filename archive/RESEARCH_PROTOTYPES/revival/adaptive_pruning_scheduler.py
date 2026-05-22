import torch

class AdaptivePruningScheduler:
    """
    Adjusts pruning aggressiveness based on attention entropy and cache pressure.
    """

    def __init__(self, target_vram_gb=None, min_keep_ratio=0.1):
        self.target_vram_gb = target_vram_gb
        self.min_keep_ratio = min_keep_ratio
        self.current_keep_ratio = 1.0

    def calculate_keep_ratio(self, attention_entropy, cache_pressure):
        """
        Signals used:
        - attention_entropy: Higher entropy means attention is more spread out, 
          making pruning riskier (keep more).
        - cache_pressure: 0 to 1 scale. 1 means cache is full (prune more).
        """
        # Linear relationship for demonstration
        # Base keep ratio influenced by cache pressure
        base_ratio = 1.0 - (cache_pressure * 0.8) 
        
        # Adjust based on entropy (0 to 1, where 1 is high entropy/flat attention)
        # If entropy is high, we want to keep more tokens to avoid losing information.
        adjustment = attention_entropy * 0.5
        
        self.current_keep_ratio = max(self.min_keep_ratio, min(1.0, base_ratio + adjustment))
        return self.current_keep_ratio

if __name__ == "__main__":
    scheduler = AdaptivePruningScheduler()
    # High pressure, low entropy (sharp attention) -> Prune aggressively
    ratio1 = scheduler.calculate_keep_ratio(attention_entropy=0.1, cache_pressure=0.9)
    # Low pressure, high entropy (flat attention) -> Keep more
    ratio2 = scheduler.calculate_keep_ratio(attention_entropy=0.8, cache_pressure=0.2)
    
    print(f"Aggressive Ratio: {ratio1:.4f}")
    print(f"Conservative Ratio: {ratio2:.4f}")
