from dataclasses import dataclass, field
from typing import List, Optional
import editdistance  # pip install editdistance

@dataclass
class ContinuationSnapshot:
    step: int
    generated_so_far: str
    prefix_ok: bool         # does generated text start with expected prefix?
    suffix_ok: bool         # does generated text end with expected suffix?
    edit_distance: int      # Levenshtein distance from expected needle
    drift_score: float      # normalised drift: edit_distance / max(len(expected), 1)

class SymbolicContinuationStabilityTracker:
    """
    PHASE 20.5: ALFSR - Symbolic Continuation Stability Tracker.

    Measures whether symbolic sequences continue correctly *after* steering weakens.

    TRACKS:
    - Prefix stability  (first K chars of expected needle present)
    - Suffix stability  (last K chars of expected needle present)
    - Drift onset       (step at which edit-distance first exceeds threshold)
    - Divergence distance (Levenshtein at end of generation)
    """

    PREFIX_CHECK_LEN = 6   # chars
    SUFFIX_CHECK_LEN = 6
    DRIFT_THRESHOLD  = 0.4 # normalised edit distance above which we flag drift

    def __init__(self, expected_needle: str):
        self.expected       = expected_needle.strip()
        self.prefix         = self.expected[:self.PREFIX_CHECK_LEN].lower()
        self.suffix         = self.expected[-self.SUFFIX_CHECK_LEN:].lower()
        self.log: List[ContinuationSnapshot] = []
        self.drift_onset: Optional[int] = None

    def record(self, generated_tokens_so_far: List[int], tokenizer) -> ContinuationSnapshot:
        """Record a step of generation."""
        text = tokenizer.decode(generated_tokens_so_far, skip_special_tokens=True).lower()
        step = len(self.log) + 1

        prefix_ok = text.startswith(self.prefix) or (self.prefix in text[:len(self.prefix) + 20])
        suffix_ok = text.endswith(self.suffix) or (self.suffix in text[-len(self.suffix) - 20:])

        # Levenshtein distance on the current output vs full expected
        ed = editdistance.eval(text[: len(self.expected) + 10], self.expected)
        drift = ed / max(len(self.expected), 1)

        if drift > self.DRIFT_THRESHOLD and self.drift_onset is None:
            self.drift_onset = step

        snap = ContinuationSnapshot(
            step=step,
            generated_so_far=text,
            prefix_ok=prefix_ok,
            suffix_ok=suffix_ok,
            edit_distance=ed,
            drift_score=drift,
        )
        self.log.append(snap)
        return snap

    def final_edit_distance(self) -> int:
        if not self.log:
            return len(self.expected)
        return self.log[-1].edit_distance

    def export(self) -> List[dict]:
        return [
            {
                "step": s.step,
                "generated_so_far": s.generated_so_far,
                "prefix_ok": s.prefix_ok,
                "suffix_ok": s.suffix_ok,
                "edit_distance": s.edit_distance,
                "drift_score": s.drift_score,
            }
            for s in self.log
        ]

    def summary(self) -> dict:
        final = self.log[-1] if self.log else None
        return {
            "expected": self.expected,
            "drift_onset_step": self.drift_onset,
            "final_edit_distance": final.edit_distance if final else -1,
            "final_drift_score": final.drift_score if final else -1.0,
            "final_prefix_ok": final.prefix_ok if final else False,
            "final_suffix_ok": final.suffix_ok if final else False,
        }
