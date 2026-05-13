import time
from typing import List, Dict, Any

class RetrievalStateJournal:
    """
    Journals retrieval events to maintain a transparent log of state transitions.
    Used for reset-safe restoration and auditing.
    """
    def __init__(self):
        self.journal: List[Dict[str, Any]] = []

    def log_retrieval(self, query_hash: str, retrieved_keys: List[int], efficiency: float):
        """Logs a retrieval event."""
        entry = {
            "timestamp": time.time(),
            "query_hash": query_hash,
            "retrieved_keys_count": len(retrieved_keys),
            "efficiency": efficiency
        }
        self.journal.append(entry)

    def get_journal_summary(self) -> Dict[str, Any]:
        """Returns a summary of retrieval activity."""
        if not self.journal:
            return {"status": "empty"}
        
        avg_efficiency = sum(e["efficiency"] for e in self.journal) / len(self.journal)
        return {
            "total_events": len(self.journal),
            "avg_efficiency": avg_efficiency,
            "last_event_timestamp": self.journal[-1]["timestamp"]
        }

    def clear(self):
        """Clears the journal."""
        self.journal = []
