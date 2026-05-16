from typing import List, Tuple
from .structural_anchor_detector import StructuralAnchorDetector

class AnchorRelativeCapsuleManager:
    """
    PHASE 18.9B: Anchor-Relative Reinforcement.
    Stabilizes symbolic regions by anchoring them to structural markers.
    Ensures that lead-in prefixes are captured before they are pruned.
    """
    def __init__(self, anchor_detector: StructuralAnchorDetector):
        self.detector = anchor_detector

    def apply_reinforcement(self, input_ids, symbolic_spans: List[Tuple[int, int]], lead_in_buffer=16):
        """
        Adjusts symbolic spans to align with nearby structural anchors.
        MANDATORY: Preserve preceding lead-in continuity.
        """
        anchor_indices = self.detector.get_anchor_indices(input_ids)
        if len(anchor_indices) == 0:
            return symbolic_spans

        reinforced_spans = []
        for start, end in symbolic_spans:
            # 1. Find the closest anchor BEFORE the start of the symbolic span
            # This is critical for preserving lead-in/prefixes
            prior_anchors = anchor_indices[anchor_indices <= start]
            
            new_start = start
            if len(prior_anchors) > 0:
                closest_anchor = prior_anchors[-1].item()
                # If the anchor is within a reasonable distance, snap to it
                # This ensures structural integrity of the symbolic block
                if start - closest_anchor < 48:
                    new_start = closest_anchor
                else:
                    # Otherwise, provide a default lead-in buffer
                    new_start = max(0, start - lead_in_buffer)
            else:
                new_start = max(0, start - lead_in_buffer)

            # 2. Find the closest anchor AFTER the end of the symbolic span
            # This helps preserve delimiters or suffixes
            following_anchors = anchor_indices[anchor_indices >= end]
            new_end = end
            if len(following_anchors) > 0:
                closest_following = following_anchors[0].item()
                if closest_following - end < 16:
                    new_end = closest_following
            
            reinforced_spans.append((new_start, new_end))
            
        return reinforced_spans
