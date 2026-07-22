"""
agents/persistent_tool_memory.py

Manages agent tool-use history and persistent state across multi-session workflows.
Uses DKV anchors to preserve tool interaction manifolds.
"""

import torch
from typing import Dict, Any, List

class PersistentToolMemory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.tool_logs = []
        self.anchor_positions = []
        
    def log_interaction(self, tool_name: str, args: Dict[str, Any], result: str, pos: int):
        """
        Logs a tool interaction and flags its position as a high-priority semantic anchor.
        """
        interaction = {
            "tool": tool_name,
            "args": args,
            "result": result,
            "position": pos
        }
        self.tool_logs.append(interaction)
        self.anchor_positions.append(pos)
        
        if len(self.tool_logs) > self.capacity:
            self.tool_logs.pop(0)
            self.anchor_positions.pop(0)
            
    def get_high_priority_anchors(self) -> List[int]:
        """Returns sequence positions that should be prioritized for stabilization."""
        return self.anchor_positions

    def summarize_context(self) -> str:
        """Returns a compressed summary of past tool interactions."""
        summary = "Tool Interaction History:\n"
        for log in self.tool_logs[-5:]: # Last 5
            summary += f"- {log['tool']}({log['args']}) -> {log['result'][:50]}...\n"
        return summary
