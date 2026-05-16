import torch
from .persistent_relevance_tracker import PersistentRelevanceTracker

class AnticipatoryCapsuleEngine:
    """
    PHASE 18.8B: Anticipatory Capsule Activation.
    Reduces boundary loss by activating protection BEFORE degradation occurs.
    """
    def __init__(self, lookback_window=8, lookahead_window=4):
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self.relevance_tracker = PersistentRelevanceTracker()

    def detect_and_expand(self, hidden_states, input_ids, chunk_idx, base_spans):
        """
        Takes base high-entropy spans and expands them based on anticipatory signals.
        """
        # 1. Update persistent relevance
        relevance_scores = self.relevance_tracker.update_relevance(hidden_states, input_ids, chunk_idx)
        
        # 2. Identify high-relevance seeds
        # Selectivity increased to avoid chunk-wide expansion
        rel_spans = self.relevance_tracker.get_high_relevance_spans(chunk_idx, threshold=0.9)
        
        # 3. Merge and Expand
        # We want to capture the "ramp-up" of relevance
        final_spans = []
        combined_seeds = self._merge_spans(base_spans + rel_spans)
        
        for start, end in combined_seeds:
            # Expand Backward (Lookback)
            # This captures prefixes that might be low-entropy but critical for the symbolic span
            new_start = max(0, start - self.lookback_window)
            
            # Expand Forward (Lookahead)
            # This stabilizes the exit boundary
            new_end = min(hidden_states.size(1) - 1, end + self.lookahead_window)
            
            final_spans.append((new_start, new_end))
            
        return self._merge_spans(final_spans)

    def _merge_spans(self, spans):
        if not spans:
            return []
        spans.sort()
        merged = [list(spans[0])]
        for curr in spans[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1] + 1: # overlapping or adjacent
                prev[1] = max(prev[1], curr[1])
            else:
                merged.append(list(curr))
        return [tuple(m) for m in merged]
