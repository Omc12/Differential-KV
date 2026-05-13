import torch
from typing import List, Dict, Optional, Any

class StructuredSummaryMemory:
    """
    Stores explicit, structured summaries of past context.
    Strictly forbids hidden state carryover.
    Only allows: text summaries, explicit token indices, or non-latent metadata.
    """
    def __init__(self, max_summaries: int = 10):
        self.summaries: List[Dict[str, Any]] = []
        self.max_summaries = max_summaries

    def add_summary(self, text_summary: str, importance_score: float, context_range: tuple):
        """
        Adds a text-based summary of a context window.
        """
        if len(self.summaries) >= self.max_summaries:
            # Evict lowest importance
            self.summaries.sort(key=lambda x: x["importance_score"])
            self.summaries.pop(0)
            
        self.summaries.append({
            "content": text_summary,
            "importance_score": importance_score,
            "range": context_range
        })

    def get_memory_prompt(self) -> str:
        """
        Serializes summaries into a prompt segment for retrieval-augmented generation.
        """
        if not self.summaries:
            return ""
            
        header = "\n[Relevant Past Context Summaries]\n"
        body = "\n".join([f"- {s['content']}" for s in self.summaries])
        return header + body

    def clear(self):
        """
        Hard reset of memory.
        """
        self.summaries = []
