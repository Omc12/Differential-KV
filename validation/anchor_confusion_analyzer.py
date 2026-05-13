"""
validation/anchor_confusion_analyzer.py

Phase 12.5D: Anchor Confusion Analyzer
Tracks and analyzes cases where the sparse retrieval system pulls the wrong
semantic anchor due to spatial or semantic proximity.
"""

from typing import Dict, List, Any

class AnchorConfusionAnalyzer:
    """
    Logs and computes statistics on retrieval errors (confusion matrix style).
    """
    def __init__(self):
        self.errors = []

    def log_error(self, expected_reason: str, retrieved_reason: str, query: str):
        self.errors.append({
            "expected": expected_reason,
            "retrieved": retrieved_reason,
            "query": query
        })

    def get_confusion_stats(self) -> Dict[str, Any]:
        if not self.errors:
            return {"total_errors": 0, "most_common_confusion": None}

        confusion_counts = {}
        for err in self.errors:
            key = f"{err['expected']} -> {err['retrieved']}"
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

        most_common = max(confusion_counts.items(), key=lambda x: x[1])

        return {
            "total_errors": len(self.errors),
            "unique_error_types": len(confusion_counts),
            "most_common_confusion": most_common[0],
            "most_common_count": most_common[1]
        }
