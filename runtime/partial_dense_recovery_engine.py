"""
RCO-N Phase 41.1: Partial Dense Recovery Engine.

Replaces full dense fallback with localized semantic repair.
Densifies ONLY unstable layers / unstable heads, not the entire sequence.

Dense fallback currently destroys acceleration.
Microburst targeted repair preserves sparse continuity.
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class RecoveryScope(IntEnum):
    NONE     = 0   # No recovery needed
    HEAD     = 1   # Single attention head repair
    LAYER    = 2   # Single layer densification
    WINDOW   = 3   # Local token window re-attend
    FULL     = 4   # Full dense fallback (last resort)


@dataclass
class PartialRecoveryPlan:
    """A targeted recovery plan for a specific semantic instability event."""
    session_id: str
    scope: RecoveryScope
    target_layers: List[int]     # Which layers to densify
    target_heads: List[int]      # Which heads to densify (empty = all heads in layer)
    window_start: int = 0        # Token window start for local re-attend
    window_end: int = -1         # Token window end (-1 = current)
    priority: int = 1            # Higher = more urgent
    created_ts: float = field(default_factory=time.perf_counter)
    reason: str = "semantic_drift"

    @property
    def scope_name(self) -> str:
        return self.scope.name

    def estimated_cost_ratio(self) -> float:
        """Rough estimate of compute cost vs full dense (0.0–1.0)."""
        if self.scope == RecoveryScope.NONE:
            return 0.0
        if self.scope == RecoveryScope.HEAD:
            return 0.05   # ~5% of full dense
        if self.scope == RecoveryScope.LAYER:
            return max(0.05, len(self.target_layers) / 28)
        if self.scope == RecoveryScope.WINDOW:
            return 0.20
        return 1.0   # FULL


class PartialDenseRecoveryEngine:
    """
    RCO-N Phase 41.1: Localized semantic repair engine.

    Key behaviors:
    1. Analyzes instability signals to identify MINIMUM recovery scope
    2. Constructs targeted recovery plans (head-level, layer-level, window-level)
    3. Executes microburst dense repair for unstable layers only
    4. Falls back to full dense only as absolute last resort (>3 failed micro-repairs)
    5. Tracks cost savings vs naive full-dense fallback
    """

    # Thresholds for scope escalation
    HEAD_DRIFT_THRESHOLD    = 0.15   # Single-head KL divergence
    LAYER_DRIFT_THRESHOLD   = 0.30   # Layer-mean drift
    WINDOW_DRIFT_THRESHOLD  = 0.50   # Session-wide drift
    FULL_DRIFT_THRESHOLD    = 0.80   # Catastrophic — full dense required

    MAX_MICRO_REPAIRS_BEFORE_FULL = 3  # After this many micro-repair failures → full dense

    def __init__(self, total_layers: int = 28, total_heads: int = 16, trace_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._logger = logging.getLogger("RCO_PartialDenseRecovery")
        self._total_layers = total_layers
        self._total_heads = total_heads

        # Per-session repair history
        self._repair_failures: Dict[str, int] = {}   # session_id -> consecutive micro failures
        self._repair_history: Dict[str, List[PartialRecoveryPlan]] = {}

        # Statistics
        self._plans_created = 0
        self._scope_counts: Dict[str, int] = {s.name: 0 for s in RecoveryScope}
        self._full_dense_prevented = 0    # Times we avoided full dense
        self._full_dense_executed = 0     # Times full dense was unavoidable
        self._cost_ratio_sum = 0.0        # Cumulative cost ratio
        self._full_dense_baseline_sum = 0 # If we had done full dense every time

        self._trace_path = Path(trace_dir) / "partial_dense_recovery_trace.jsonl" if trace_dir else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "PartialDenseRecoveryEngine initialized | "
            "layers=%d heads=%d | micro-repair threshold=%d",
            total_layers, total_heads, self.MAX_MICRO_REPAIRS_BEFORE_FULL
        )

    # -----------------------------------------------------------------------
    # Plan construction (analysis → scope decision)
    # -----------------------------------------------------------------------

    def construct_recovery_plan(
        self,
        session_id: str,
        layer_drift_scores: Dict[int, float],    # layer_idx -> drift score
        head_drift_scores: Dict[Tuple[int,int], float] = None,  # (layer, head) -> drift
        continuity_score: float = 1.0,
    ) -> PartialRecoveryPlan:
        """
        Given instability signals, construct the MINIMUM recovery plan.
        Prefer head-level > layer-level > window > full dense.
        """
        consecutive_failures = self._repair_failures.get(session_id, 0)

        # Escalate to full if too many micro-repair failures
        if consecutive_failures >= self.MAX_MICRO_REPAIRS_BEFORE_FULL:
            self._repair_failures[session_id] = 0
            return self._plan(session_id, RecoveryScope.FULL, [], [], reason="max_micro_failures")

        # Identify unstable layers
        unstable_layers = [
            l for l, drift in layer_drift_scores.items()
            if drift > self.LAYER_DRIFT_THRESHOLD
        ]

        # Identify unstable heads (if head-level data available)
        unstable_heads_by_layer: Dict[int, List[int]] = {}
        if head_drift_scores:
            for (layer, head), drift in head_drift_scores.items():
                if drift > self.HEAD_DRIFT_THRESHOLD:
                    if layer not in unstable_heads_by_layer:
                        unstable_heads_by_layer[layer] = []
                    unstable_heads_by_layer[layer].append(head)

        # Decision logic: smallest effective scope first
        if not unstable_layers:
            # Possibly just continuity degradation
            if continuity_score < 0.6:
                return self._plan(session_id, RecoveryScope.WINDOW, [], [], reason="continuity_low")
            return self._plan(session_id, RecoveryScope.NONE, [], [], reason="no_action")

        # Check if head-level repair is sufficient
        if (unstable_heads_by_layer and
                all(len(h) <= self._total_heads // 4 for h in unstable_heads_by_layer.values()) and
                len(unstable_layers) <= 3):
            # Head-level repair: only a subset of heads in specific layers
            flat_heads = []
            for heads in unstable_heads_by_layer.values():
                flat_heads.extend(heads)
            return self._plan(
                session_id, RecoveryScope.HEAD, unstable_layers, flat_heads,
                reason="head_drift"
            )

        # Layer-level repair: if < 50% of layers are unstable
        if len(unstable_layers) <= self._total_layers // 2:
            return self._plan(session_id, RecoveryScope.LAYER, unstable_layers, [], reason="layer_drift")

        # Window-level repair: most layers are drifting — re-attend local window
        if continuity_score > 0.3:
            return self._plan(session_id, RecoveryScope.WINDOW, unstable_layers, [], reason="window_drift")

        # Full dense: last resort
        return self._plan(session_id, RecoveryScope.FULL, [], [], reason="catastrophic_drift")

    def _plan(
        self, session_id: str, scope: RecoveryScope,
        layers: List[int], heads: List[int], reason: str = ""
    ) -> PartialRecoveryPlan:
        plan = PartialRecoveryPlan(
            session_id=session_id,
            scope=scope,
            target_layers=layers,
            target_heads=heads,
            reason=reason,
        )
        with self._lock:
            self._plans_created += 1
            self._scope_counts[scope.name] += 1

            if scope == RecoveryScope.FULL:
                self._full_dense_executed += 1
            elif scope != RecoveryScope.NONE:
                self._full_dense_prevented += 1

            self._cost_ratio_sum += plan.estimated_cost_ratio()
            self._full_dense_baseline_sum += 1.0

        self._persist_plan(plan)
        return plan

    # -----------------------------------------------------------------------
    # Repair execution (hook for actual model surgery)
    # -----------------------------------------------------------------------

    def execute_plan(self, plan: PartialRecoveryPlan, model_context: Any = None) -> bool:
        """
        Execute a recovery plan. For HEAD/LAYER/WINDOW, calls targeted
        densification hooks. Returns True on success.

        In the current implementation, this returns True for any scope < FULL
        and logs the operation. Real model surgery is injected via the
        model_context hooks (provided by hf_diffkv_wrapper).
        """
        if plan.scope == RecoveryScope.NONE:
            return True

        t0 = time.perf_counter()
        success = False

        try:
            if plan.scope == RecoveryScope.HEAD:
                success = self._execute_head_repair(plan, model_context)
            elif plan.scope == RecoveryScope.LAYER:
                success = self._execute_layer_repair(plan, model_context)
            elif plan.scope == RecoveryScope.WINDOW:
                success = self._execute_window_repair(plan, model_context)
            elif plan.scope == RecoveryScope.FULL:
                success = self._execute_full_dense(plan, model_context)
        except Exception as e:
            self._logger.warning("Recovery execution error: %s", e)
            success = False

        elapsed = time.perf_counter() - t0

        if not success and plan.scope != RecoveryScope.FULL:
            with self._lock:
                self._repair_failures[plan.session_id] = (
                    self._repair_failures.get(plan.session_id, 0) + 1
                )
        elif success:
            with self._lock:
                self._repair_failures[plan.session_id] = 0

        self._logger.debug(
            "Recovery [%s] session=%s scope=%s layers=%s success=%s %.1fms",
            "OK" if success else "FAIL",
            plan.session_id, plan.scope_name,
            plan.target_layers[:4], success,
            elapsed * 1000
        )
        return success

    def _execute_head_repair(self, plan: PartialRecoveryPlan, ctx: Any) -> bool:
        """Dense re-compute for specific attention heads only."""
        if ctx and hasattr(ctx, "densify_heads"):
            return ctx.densify_heads(plan.target_layers, plan.target_heads)
        return True   # Structural success (hooks not yet injected)

    def _execute_layer_repair(self, plan: PartialRecoveryPlan, ctx: Any) -> bool:
        """Dense re-compute for specific layers only."""
        if ctx and hasattr(ctx, "densify_layers"):
            return ctx.densify_layers(plan.target_layers)
        return True

    def _execute_window_repair(self, plan: PartialRecoveryPlan, ctx: Any) -> bool:
        """Local window re-attention for continuity recovery."""
        if ctx and hasattr(ctx, "reattend_window"):
            return ctx.reattend_window(plan.window_start, plan.window_end)
        return True

    def _execute_full_dense(self, plan: PartialRecoveryPlan, ctx: Any) -> bool:
        """Full dense pass — last resort."""
        if ctx and hasattr(ctx, "full_dense_pass"):
            return ctx.full_dense_pass()
        return True

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_recovery_stats(self) -> Dict[str, Any]:
        total = max(self._plans_created, 1)
        avg_cost = round(self._cost_ratio_sum / total, 4)
        baseline_cost = self._full_dense_baseline_sum
        cost_savings_pct = round(
            (1.0 - self._cost_ratio_sum / max(baseline_cost, 1)) * 100, 1
        )
        return {
            "plans_created": self._plans_created,
            "scope_counts": dict(self._scope_counts),
            "full_dense_prevented": self._full_dense_prevented,
            "full_dense_executed": self._full_dense_executed,
            "avg_cost_ratio": avg_cost,
            "cost_savings_vs_full_dense_pct": cost_savings_pct,
        }

    def format_live_line(self) -> str:
        s = self.get_recovery_stats()
        return (
            f"[PARTIAL_RECOVERY] plans={s['plans_created']} "
            f"full_prevented={s['full_dense_prevented']} "
            f"full_executed={s['full_dense_executed']} "
            f"cost_savings={s['cost_savings_vs_full_dense_pct']:.1f}%"
        )

    def _persist_plan(self, plan: PartialRecoveryPlan):
        if not self._trace_path:
            return
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "session_id": plan.session_id,
                    "scope": plan.scope_name,
                    "target_layers": plan.target_layers[:8],
                    "target_heads": plan.target_heads[:8],
                    "estimated_cost_ratio": plan.estimated_cost_ratio(),
                    "reason": plan.reason,
                }) + "\n")
        except Exception:
            pass
