from typing import List
from benchmark_mode_classifier import BenchmarkMode
from benchmark_component_registry import registry
from telemetry_scope_tracker import scope_tracker

class BenchmarkTruthValidator:
    """
    Validates that reported metrics are consistent with the declared mode.
    """
    def __init__(self, mode: BenchmarkMode):
        self.mode = mode
        self.violations = []

    def validate(self):
        manifest = registry.get_participation_manifest()
        scope = scope_tracker.get_scope_manifest()

        if self.mode == BenchmarkMode.PRODUCTION:
            # Production must include everything
            required = {"tokenizer", "logits", "sampling", "embeddings"}
            missing = required - set(manifest)
            if missing:
                self.violations.append(f"PRODUCTION mode missing required components: {missing}")
            if not scope["model_weights"]:
                self.violations.append("PRODUCTION mode VRAM must include model weights")
            if not scope["wall_clock"]:
                self.violations.append("PRODUCTION mode must use real wall-clock timing")

        elif self.mode == BenchmarkMode.INTEGRATED:
            # Integrated must include real model weights and logits
            if "logits" not in manifest:
                self.violations.append("INTEGRATED mode must include real logits")
            if not scope["model_weights"]:
                self.violations.append("INTEGRATED mode must include model weights in VRAM")

        elif self.mode == BenchmarkMode.SUBSYSTEM:
            # Subsystem should NOT claim production performance
            if "tokenizer" in manifest and "logits" in manifest:
                 pass # This is fine, but we must ensure it's not reported as production TPS

    def get_violations(self) -> List[str]:
        return self.violations
