import torch
from typing import List, Dict, Optional
import json

class HierarchicalSummaryMemory:
    """
    Hierarchical summary memory for long-context continuity.
    Stores explicit text/token summaries at multiple levels.
    Strictly NO hidden activations are persisted.
    """
    def __init__(self, block_size: int = 1024, summary_levels: int = 2):
        self.block_size = block_size
        self.summary_levels = summary_levels
        # Level 0: Raw blocks (if needed for context)
        # Level 1: Block summaries
        # Level 2: Higher-level summaries (e.g., summary of level 1 summaries)
        self.memory: Dict[int, List[str]] = {i: [] for i in range(1, summary_levels + 1)}
        self.current_block_tokens: List[str] = []
        
    def add_tokens(self, tokens: List[str]):
        """Add new tokens and trigger summarization if block is full."""
        self.current_block_tokens.extend(tokens)
        while len(self.current_block_tokens) >= self.block_size:
            block = self.current_block_tokens[:self.block_size]
            self.current_block_tokens = self.current_block_tokens[self.block_size:]
            self._summarize_block(block)

    def _summarize_block(self, block: List[str]):
        """Generate a summary for a block and propagate up the hierarchy."""
        # In a real implementation, this would call a model.
        # For now, we simulate a summary (e.g., take first 10% of tokens as 'key points').
        # This is a placeholder for a production summarization call.
        summary_text = f"[Summary of block {len(self.memory[1])}: {' '.join(block[:20])}...]"
        self.memory[1].append(summary_text)
        
        # Check if we need to summarize at level 2
        if len(self.memory[1]) % 10 == 0 and self.summary_levels >= 2:
            level_1_chunk = self.memory[1][-10:]
            higher_summary = f"[Higher Summary: {level_1_chunk[0][:30]}...]"
            self.memory[2].append(higher_summary)

    def get_context_bridge(self, query: str, max_tokens: int = 2048) -> str:
        """
        Construct a retrieval bridge prompt using hierarchical summaries.
        Returns a string to be prepended to the current context.
        """
        # Simple retrieval: return the most recent summaries or use the query to filter.
        # For now, return the most recent level 1 summaries that fit in max_tokens.
        bridge = []
        current_len = 0
        
        # Prioritize higher-level summaries first for broad context
        if self.summary_levels >= 2:
            for s in reversed(self.memory[2]):
                if current_len + len(s) < max_tokens // 4:
                    bridge.append(s)
                    current_len += len(s)
        
        # Then fill with level 1 summaries
        for s in reversed(self.memory[1]):
            if current_len + len(s) < max_tokens:
                bridge.append(s)
                current_len += len(s)
                
        return "\n".join(reversed(bridge))

    def reset(self):
        """Hard reset the memory to prevent leakage."""
        for i in self.memory:
            self.memory[i] = []
        self.current_block_tokens = []

    def export_state(self) -> str:
        """Serialize memory for persistence."""
        return json.dumps(self.memory)

    def import_state(self, state_json: str):
        """Restore memory from serialized state."""
        self.memory = {int(k): v for k, v in json.loads(state_json).items()}
