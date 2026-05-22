class SymbolicTransitionStabilizer:
    """
    PHASE 18.9C: Symbolic Transition Stabilizer.
    Smoothes the importance gradient at the transitions of symbolic spans.
    """
    def __init__(self, transition_window=8):
        self.transition_window = transition_window

    def apply_smoothing(self, importance_scores, symbolic_mask):
        import torch
        # importance_scores: [batch, seq_len]
        # symbolic_mask: [batch, seq_len]
        
        smoothed = importance_scores.clone()
        # Find where symbolic mask transitions from 0 to 1 or 1 to 0
        diff = symbolic_mask[:, 1:].float() - symbolic_mask[:, :-1].float()
        transitions = torch.where(diff != 0)
        
        for b, idx in zip(transitions[0], transitions[1]):
            # Apply a linear ramp around the transition
            start = max(0, idx - self.transition_window)
            end = min(importance_scores.size(1), idx + self.transition_window)
            for i in range(start, end):
                dist = abs(i - idx)
                factor = 1.0 - (dist / self.transition_window)
                # Boost importance near transition
                smoothed[b, i] = max(smoothed[b, i], importance_scores[b, i] * (1.0 + factor))
                
        return smoothed
