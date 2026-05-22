"""
analysis/pivot_detector.py
Phase 15: Reasoning Pivot Detection
Identifies critical points in Chain-of-Thought and logical transitions.
"""

import torch
from typing import List, Dict, Any

class ReasoningPivotDetector:
    def __init__(self):
        self.pivots = []
        # Key patterns that signal logical branching or state transitions
        self.logical_anchors = {
            "causal": ["because", "therefore", "thus", "consequently", "so"],
            "sequential": ["first", "second", "finally", "next", "step"],
            "conditional": ["if", "unless", "suppose", "assume"],
            "terminal": ["answer", "result", "conclusion"]
        }

    def detect_pivot(self, tokens: List[int], tokenizer, attention_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detects if the current token is a reasoning pivot.
        Uses both semantic (token) and mechanistic (attention) signals.
        """
        if not tokens: return {"is_pivot": False}
        
        last_token = tokenizer.decode([tokens[-1]]).strip().lower()
        
        # Signal 1: Semantic Pivot
        semantic_type = None
        for ptype, words in self.logical_anchors.items():
            if last_token in words:
                semantic_type = ptype
                break
        
        # Signal 2: Mechanistic Pivot (Attention Entropy Spikes)
        # Pivots often show a reorganization of attention
        is_mechanistic_pivot = attention_metrics.get("attention_fragmentation", 0) > 0.4
        
        is_pivot = (semantic_type is not None) or is_mechanistic_pivot
        
        pivot_info = {
            "is_pivot": is_pivot,
            "type": semantic_type or ("mechanistic" if is_mechanistic_pivot else None),
            "risk_level": "high" if is_pivot else "low"
        }
        
        if is_pivot:
            self.pivots.append(pivot_info)
            
        return pivot_info

    def get_pivot_map(self):
        return self.pivots

if __name__ == "__main__":
    detector = ReasoningPivotDetector()
    # Mock tokens and metrics
    class MockTokenizer:
        def decode(self, ids): return "therefore" if ids[0] == 1 else "cat"
    
    tokenizer = MockTokenizer()
    print("Pivot Check (therefore):", detector.detect_pivot([1], tokenizer, {"attention_fragmentation": 0.1}))
    print("Pivot Check (cat, high frag):", detector.detect_pivot([2], tokenizer, {"attention_fragmentation": 0.6}))
