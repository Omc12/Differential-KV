import torch
from dataclasses import dataclass
from typing import List

@dataclass
class FreedomSnapshot:
    step: int
    entropy_nats: float
    top1_prob: float
    top5_mass: float
    unique_topk: int         # unique tokens in top-20
    is_deterministic: bool   # True if top-1 prob > 0.95 (forced-feeling)
    is_looping: bool         # True if same token repeated 3+ times
    classification: str      # "probabilistic" | "guided" | "forced"

class ProbabilisticFreedomAuditor:
    """
    PHASE 20.5: ALFSR - Probabilistic Freedom Auditor.

    Verifies that retrieval remains:
    - Probabilistic (entropy > floor)
    - Generative    (no repeated forced outputs)
    - Flexible      (top-k spread remains healthy)

    DETECTS:
    - Deterministic replay  (top1 > 0.95)
    - Collapsed beam        (same token repeated)
    - Entropy floor collapse (entropy < 0.5 nats)
    - Forced outputs         (consecutive deterministic steps)

    CLASSIFICATION (per step):
    - "probabilistic" : entropy healthy, no repetition, spread OK
    - "guided"        : entropy borderline, mild top-1 dominance
    - "forced"        : deterministic, repeated, or collapsed
    """

    ENTROPY_FLOOR     = 0.5
    DETERMINISM_PROB  = 0.90   # top-1 prob threshold
    GUIDED_TOP1       = 0.70

    def __init__(self, top_k_track: int = 20):
        self.top_k = top_k_track
        self.log: List[FreedomSnapshot] = []
        self.recent_tokens: List[int] = []
        self.step = 0

    def record(self, logits: torch.Tensor, generated_token_id: int) -> FreedomSnapshot:
        self.step += 1
        self.recent_tokens.append(generated_token_id)

        with torch.no_grad():
            probs = torch.softmax(logits.float().squeeze(0), dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum().item()
            top1_prob = probs.max().item()
            top5_mass = probs.topk(5).values.sum().item()

            topk_indices = probs.topk(self.top_k).indices
            unique_topk = len(set(topk_indices.tolist()))

        # Looping: same token 3 times in a row
        is_looping = (
            len(self.recent_tokens) >= 3
            and len(set(self.recent_tokens[-3:])) == 1
        )

        is_deterministic = top1_prob > self.DETERMINISM_PROB

        # Classify
        if is_deterministic or is_looping or entropy < self.ENTROPY_FLOOR:
            classification = "forced"
        elif top1_prob > self.GUIDED_TOP1 or entropy < 2.0:
            classification = "guided"
        else:
            classification = "probabilistic"

        snap = FreedomSnapshot(
            step=self.step,
            entropy_nats=entropy,
            top1_prob=top1_prob,
            top5_mass=top5_mass,
            unique_topk=unique_topk,
            is_deterministic=is_deterministic,
            is_looping=is_looping,
            classification=classification,
        )
        self.log.append(snap)
        return snap

    def forced_fraction(self) -> float:
        if not self.log:
            return 0.0
        return sum(1 for s in self.log if s.classification == "forced") / len(self.log)

    def probabilistic_fraction(self) -> float:
        if not self.log:
            return 0.0
        return sum(1 for s in self.log if s.classification == "probabilistic") / len(self.log)

    def export(self) -> List[dict]:
        return [
            {
                "step": s.step,
                "entropy_nats": s.entropy_nats,
                "top1_prob": s.top1_prob,
                "top5_mass": s.top5_mass,
                "unique_topk": s.unique_topk,
                "is_deterministic": s.is_deterministic,
                "is_looping": s.is_looping,
                "classification": s.classification,
            }
            for s in self.log
        ]

    def summary(self) -> dict:
        return {
            "total_steps": self.step,
            "probabilistic_fraction": self.probabilistic_fraction(),
            "guided_fraction": sum(1 for s in self.log if s.classification == "guided") / max(self.step, 1),
            "forced_fraction": self.forced_fraction(),
            "mean_entropy": sum(s.entropy_nats for s in self.log) / max(self.step, 1),
            "mean_top1_prob": sum(s.top1_prob for s in self.log) / max(self.step, 1),
            "looping_detected": any(s.is_looping for s in self.log),
        }

    def reset(self):
        self.log.clear()
        self.recent_tokens.clear()
        self.step = 0
