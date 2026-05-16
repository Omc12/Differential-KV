import torch

class QueryRelevancePredictor:
    """
    PHASE 20.1B: Predicts if a token is likely to be relevant for future queries.
    Uses structural cues (assignment operators, colons) to boost importance.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Indicators of assignment or definition (domain-general)
        self.query_anchors = ["=", ":", "constant", "key", "id", "code", "\u662f", "\uff1a"] 
        self.anchor_ids = [tokenizer.encode(a, add_special_tokens=False)[0] for a in self.query_anchors if len(tokenizer.encode(a, add_special_tokens=False)) > 0]
        self.anchor_bonus = 2.0

    def predict_relevance(self, input_ids: torch.Tensor, salience_scores: torch.Tensor) -> torch.Tensor:
        """
        Boosts importance of tokens near query anchors.
        """
        batch, q_len = input_ids.shape
        relevance_boost = torch.ones_like(salience_scores)
        
        # Identify anchors
        anchor_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for aid in self.anchor_ids:
            anchor_mask |= (input_ids == aid)
            
        # Spread relevance (tokens near anchors are often the targets)
        # Simple window-based spread
        for b in range(batch):
            anchors = anchor_mask[b].nonzero().flatten()
            for a_idx in anchors:
                start = max(0, a_idx - 5)
                end = min(q_len, a_idx + 15)
                relevance_boost[b, start:end] += self.anchor_bonus
                
        return relevance_boost
