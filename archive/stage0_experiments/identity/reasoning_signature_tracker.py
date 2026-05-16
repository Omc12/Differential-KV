import torch
import numpy as np
from typing import Dict, List, Any
from collections import deque

class ReasoningSignatureTracker:
    """
    Tracks the evolution of reasoning styles and behavioral signatures.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.entropy_history = deque(maxlen=window_size)
        self.resonance_history = deque(maxlen=window_size)
        self.drift_history = deque(maxlen=window_size)
        self.style_signatures = {}

    def update(self, metrics: Dict[str, float]):
        """
        Updates the history with current reasoning metrics.
        """
        if 'entropy' in metrics:
            self.entropy_history.append(metrics['entropy'])
        if 'resonance' in metrics:
            self.resonance_history.append(metrics['resonance'])
        if 'drift' in metrics:
            self.drift_history.append(metrics['drift'])

    def compute_signature(self) -> Dict[str, Any]:
        """
        Computes the current reasoning signature based on history.
        """
        if not self.entropy_history:
            return {}
            
        signature = {
            "avg_entropy": np.mean(self.entropy_history),
            "entropy_variance": np.var(self.entropy_history),
            "avg_resonance": np.mean(self.resonance_history) if self.resonance_history else 0,
            "resonance_stability": 1.0 - np.var(self.resonance_history) if self.resonance_history else 1.0,
            "drift_velocity": np.mean(np.diff(list(self.drift_history))) if len(self.drift_history) > 1 else 0
        }
        return signature

    def detect_style_shift(self, threshold: float = 0.2) -> bool:
        """
        Detects if the reasoning style has significantly shifted.
        """
        if len(self.entropy_history) < self.window_size:
            return False
            
        first_half = list(self.entropy_history)[:self.window_size//2]
        second_half = list(self.entropy_history)[self.window_size//2:]
        
        diff = abs(np.mean(first_half) - np.mean(second_half))
        return diff > threshold

    def archive_signature(self, label: str):
        """
        Archives the current signature with a label (e.g., 'math', 'coding').
        """
        self.style_signatures[label] = self.compute_signature()
