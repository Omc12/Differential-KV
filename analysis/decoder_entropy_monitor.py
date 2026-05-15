import torch
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List

@dataclass
class EntropySnapshot:
    step: int
    entropy_nats: float
    top1_prob: float
    top5_mass: float
    repetition_prob: float  # P(last token repeats)

class DecoderEntropyMonitor:
    """
    PHASE 20.5: ALFSR - Decoder Entropy Monitor.
    
    Measures whether steering is collapsing:
    - Token diversity
    - Shannon entropy of the distribution
    - Probabilistic flexibility (top-k spread)
    - Repetition probability

    Detects oversteering by flagging sustained entropy drops.
    """

    COLLAPSE_ENTROPY_THRESHOLD = 0.5   # nats — below this = collapsed
    REPETITION_THRESHOLD       = 0.85  # if P(last token) > this = looping

    def __init__(self, history_window: int = 16):
        self.history: deque[EntropySnapshot] = deque(maxlen=history_window)
        self.collapse_events: List[int] = []
        self.step = 0

    def record(self, logits: torch.Tensor, last_token_id: int = -1) -> EntropySnapshot:
        """
        Record a single decoding step.
        
        Args:
            logits: raw logits tensor of shape [1, vocab_size] (float32 preferred).
            last_token_id: the token generated in the immediately previous step.
        
        Returns:
            EntropySnapshot for this step.
        """
        self.step += 1
        
        with torch.no_grad():
            probs = torch.softmax(logits.float().squeeze(0), dim=-1)  # [vocab]

            # Shannon entropy (nats)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum().item()

            # Top-1 probability
            top1_prob = probs.max().item()

            # Top-5 mass
            top5_mass = probs.topk(5).values.sum().item()

            # Repetition probability
            rep_prob = 0.0
            if last_token_id >= 0 and last_token_id < probs.shape[0]:
                rep_prob = probs[last_token_id].item()

        snap = EntropySnapshot(
            step=self.step,
            entropy_nats=entropy,
            top1_prob=top1_prob,
            top5_mass=top5_mass,
            repetition_prob=rep_prob,
        )
        self.history.append(snap)

        if entropy < self.COLLAPSE_ENTROPY_THRESHOLD:
            self.collapse_events.append(self.step)

        return snap

    def is_collapsed(self) -> bool:
        """True if the last 4 steps all had near-zero entropy."""
        if len(self.history) < 4:
            return False
        return all(s.entropy_nats < self.COLLAPSE_ENTROPY_THRESHOLD for s in list(self.history)[-4:])

    def is_looping(self) -> bool:
        """True if high repetition probability for 3+ consecutive steps."""
        if len(self.history) < 3:
            return False
        return all(s.repetition_prob > self.REPETITION_THRESHOLD for s in list(self.history)[-3:])

    def mean_entropy(self) -> float:
        if not self.history:
            return 0.0
        return sum(s.entropy_nats for s in self.history) / len(self.history)

    def export(self) -> List[dict]:
        return [
            {
                "step": s.step,
                "entropy_nats": s.entropy_nats,
                "top1_prob": s.top1_prob,
                "top5_mass": s.top5_mass,
                "repetition_prob": s.repetition_prob,
            }
            for s in self.history
        ]

    def reset(self):
        self.history.clear()
        self.collapse_events.clear()
        self.step = 0
