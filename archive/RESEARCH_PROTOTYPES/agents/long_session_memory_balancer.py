"""
agents/long_session_memory_balancer.py

Balances memory usage during multi-hour agent sessions.
Ensures that growing conversation history doesn't evict critical code anchors.
"""

from typing import Dict, Any, List
import logging

class LongSessionMemoryBalancer:
    """
    Budget manager for multi-session agent memory.
    """
    def __init__(self, total_budget_tokens: int):
        self.total_budget = total_budget_tokens
        self.conversation_tokens = 0
        self.code_anchor_tokens = 0
        self.logger = logging.getLogger("LongSessionMemoryBalancer")

    def allocate(self, conv_tokens: int, code_tokens: int):
        """
        Adjusts allocations to stay within total budget.
        Priority: Code Anchors > Conversation History (for technical tasks).
        """
        self.conversation_tokens = conv_tokens
        self.code_anchor_tokens = code_tokens
        
        excess = (self.conversation_tokens + self.code_anchor_tokens) - self.total_budget
        
        if excess > 0:
            # First, trim conversation history (evict oldest messages)
            self.conversation_tokens = max(1000, self.conversation_tokens - excess)
            new_excess = (self.conversation_tokens + self.code_anchor_tokens) - self.total_budget
            
            if new_excess > 0:
                # If still over, trim code anchors
                self.code_anchor_tokens -= new_excess
                self.logger.warning(f"BUDGET EXCEEDED: Trimming code anchors by {new_excess} tokens.")
            else:
                self.logger.info(f"BUDGET REBALANCED: Trimmed conversation history by {excess} tokens.")

    def get_quotas(self) -> Dict[str, int]:
        """Returns current token quotas."""
        return {
            "conversation": self.conversation_tokens,
            "code_anchors": self.code_anchor_tokens,
            "utilization": (self.conversation_tokens + self.code_anchor_tokens) / self.total_budget
        }
