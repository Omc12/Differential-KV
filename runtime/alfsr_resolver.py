import torch
import math
import time
from typing import Optional

from runtime.attention_steering_resolver import AttentionSteeringResolver
from memory.adaptive_steering_scheduler import AdaptiveSteeringScheduler
from analysis.steering_decay_controller import SteeringDecayController
from analysis.decoder_entropy_monitor import DecoderEntropyMonitor
from analysis.probabilistic_freedom_auditor import ProbabilisticFreedomAuditor

class ALFSRResolver(AttentionSteeringResolver):
    """
    PHASE 20.5: ALFSR - Adaptive Low-Force Symbolic Recovery.

    Replaces the fixed 15.0 logit bias with a fully adaptive, decaying
    steering schedule. The guidance signal:
    - Starts at `initial_strength` (configurable)
    - Scales with decoder uncertainty (entropy)
    - Decays exponentially after retrieval is confirmed
    - Approaches zero when symbolic continuation is stable

    All steering decisions are measured and exported to telemetry logs.
    """

    def __init__(
        self,
        tokenizer,
        anchor_budget: int = 6144,
        fidelity_budget: int = 1024,
        initial_strength: float = 1.0,
        max_strength: float = 8.0,
    ):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)

        # Override the fixed bias strength — ALFSR is fully adaptive
        self.logit_bias_strength = 0.0   # will be set per-step

        self.initial_strength = initial_strength
        self.max_strength = max_strength

        # ALFSR modules
        self.steering_scheduler = AdaptiveSteeringScheduler(
            base_strength=initial_strength,
            max_strength=max_strength,
            decay_rate=0.15,
        )
        self.decay_controller = SteeringDecayController(
            warmup_steps=4,
            half_life_steps=6.0,
            floor_strength=0.0,
        )
        self.entropy_monitor = DecoderEntropyMonitor(history_window=32)
        self.freedom_auditor = ProbabilisticFreedomAuditor()

        # Per-generation state
        self._last_token_id: int = -1
        self._gen_step: int = 0
        self._retrieval_confirmed: bool = False

        # Telemetry lists (appended every decoding step)
        self.steering_trace = []     # raw_strength, effective_strength, step
        self.entropy_trace  = []     # delegated to entropy_monitor.export()
        self.freedom_trace  = []     # delegated to freedom_auditor.export()

    # ------------------------------------------------------------------
    # guide_decoder override
    # ------------------------------------------------------------------

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        PHASE 20.5: Adaptive low-force steering.
        """
        self._gen_step += 1

        # --- NaN guard ---
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits)

        # --- Base calibration (DTASCC inheritance, no fixed bias) ---
        calibrated_logits = super(AttentionSteeringResolver, self).guide_decoder(logits)

        # --- Measure entropy BEFORE steering ---
        snap_pre = self.entropy_monitor.record(calibrated_logits, self._last_token_id)

        # --- Measure span attention mass ---
        span_mass = 0.0
        locked_token_ids = self.booster.get_locked_token_ids()

        if locked_token_ids and attention_weights is not None:
            try:
                if torch.isnan(attention_weights).any():
                    attention_weights = torch.nan_to_num(attention_weights)
                mass = attention_weights[-1, 0].mean(dim=0)[-1]
                current_abs_indices = self.geometry.absolute_indices[0]
                steering_bias_vec = self.booster.get_steering_bias(current_abs_indices, mass.device)
                
                # Fix: mass and steering_bias_vec must have same length
                min_len = min(mass.shape[0], steering_bias_vec.shape[0])
                mass = mass[:min_len]
                steering_bias_vec = steering_bias_vec[:min_len]
                    
                span_mass = mass[steering_bias_vec > 1.0].sum().item()
            except Exception:
                pass

        # --- Confirm retrieval ---
        if span_mass >= self.steering_scheduler.stabilization_threshold:
            self._retrieval_confirmed = True

        # --- Adaptive steering strength ---
        raw_strength = self.steering_scheduler.step_update(
            logit_entropy=snap_pre.entropy_nats,
            span_attention_mass=span_mass,
            continuation_correct=False,   # updated after token selection
        )
        effective_strength = self.decay_controller.compute(
            raw_strength=raw_strength,
            retrieval_confirmed=self._retrieval_confirmed,
        )

        # --- Apply token-space bias ---
        if locked_token_ids and effective_strength > 1e-4:
            token_indices = torch.tensor(locked_token_ids, device=calibrated_logits.device)
            token_indices = token_indices[token_indices < calibrated_logits.shape[-1]]
            if len(token_indices) > 0:
                calibrated_logits[0, token_indices] += effective_strength

        # --- Final NaN guard ---
        calibrated_logits = torch.nan_to_num(calibrated_logits)

        # --- Record telemetry ---
        self.steering_trace.append({
            "step": self._gen_step,
            "raw_strength": raw_strength,
            "effective_strength": effective_strength,
            "span_mass": span_mass,
            "retrieval_confirmed": self._retrieval_confirmed,
            "entropy_nats": snap_pre.entropy_nats,
        })

        return calibrated_logits

    def record_generated_token(self, token_id: int, logits_after: torch.Tensor):
        """Call this after each token is selected, before the next guide_decoder call."""
        self._last_token_id = token_id
        snap = self.freedom_auditor.record(logits_after, token_id)
        return snap

    def reset_generation_state(self):
        """Reset per-generation tracking (call between different generation runs)."""
        self._last_token_id = -1
        self._gen_step = 0
        self._retrieval_confirmed = False
        self.steering_scheduler.reset()
        self.decay_controller.reset()
        self.entropy_monitor.reset()
        self.freedom_auditor.reset()
        self.steering_trace.clear()
        self.entropy_trace.clear()
        self.freedom_trace.clear()

    def export_telemetry(self) -> dict:
        return {
            "steering_trace": self.steering_trace,
            "entropy_trace": self.entropy_monitor.export(),
            "freedom_summary": self.freedom_auditor.summary(),
            "freedom_trace": self.freedom_auditor.export(),
            "decay_trace": self.decay_controller.export(),
        }
