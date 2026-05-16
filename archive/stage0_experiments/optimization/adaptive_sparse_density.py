class AdaptiveSparseDensity:
    def __init__(self, base_density=0.1):
        self.base_density = base_density

    def calculate_density(self, layer_id, entropy_score, context_len):
        # Deep layers and high entropy regions get denser attention
        # Extremely long contexts decay the baseline density
        adjusted = self.base_density * (1.0 + entropy_score)
        if context_len > 65536:
            adjusted *= 0.5
        return max(0.01, min(adjusted, 1.0))
