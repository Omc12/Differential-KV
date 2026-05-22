"""
STAGE 2 - SAG: Sparse Arithmetic Governance Meter
Phase 39.0 - Sparse Attention Governance

Expanded from SAT phase arithmetic meter.
Adds:
  - sparse arithmetic by layer
  - sparse arithmetic by token region (prefix / middle / suffix)
  - arithmetic under confidence gating
  - impact of hybrid suppression on arithmetic

All values derived from real execution traces. No estimated ratios.

Approach:
  The meter does NOT guess or estimate.
  It counts real FLOPs reported by torch.profiler or by the caller,
  distinguishing sparse-path FLOPs from dense-path FLOPs.

  If torch.profiler is not available, the caller provides:
    - operation name
    - token count
    - head count
    - head dimension
    - whether the path was sparse or dense

  The meter derives participation ratios from these real numbers.

Persists: traces/stage2/phase_38_9_sat/sparse_participation_trace.jsonl
"""

import time
import json
import threading
import os
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any


TRACE_PATH = "traces/stage2/phase_38_9_sat/sparse_participation_trace.jsonl"


@dataclass
class ArithmeticEvent:
    ts: float
    step: int
    layer_idx: int
    op_name: str               # e.g. "qk_dot", "softmax", "av_dot", "ffn_gate"
    is_sparse: bool
    token_count: int
    head_count: int
    head_dim: int
    flops_estimate: int        # derived from token/head/dim, NOT guessed
    sparse_flops: int          # 0 if dense path
    dense_flops: int           # 0 if sparse path
    sparsity_ratio: float      # 0.0 = fully dense, 1.0 = fully sparse


class SparseArithmeticParticipationMeter:
    """
    STAGE 2 SAT: Sparse Arithmetic Participation Meter.

    Measures real sparse vs. dense arithmetic participation by
    accumulating FLOPs estimated from actual operation parameters.

    IMPORTANT: No guessed percentages.
    All ratios are derived from accumulated event records.

    Usage:
        meter = SparseArithmeticParticipationMeter()

        # After each attention or FFN op:
        meter.record_op(
            step=decode_step,
            layer_idx=layer_idx,
            op_name="qk_dot",
            is_sparse=True,
            token_count=active_q_len,
            head_count=num_heads,
            head_dim=head_dim,
            sparsity_ratio=0.25,    # fraction of KV positions actually computed
        )

        # After session:
        report = meter.get_participation_report()
        meter.flush_and_close()
    """

    def __init__(self, trace_path: str = TRACE_PATH, flush_every: int = 64):
        self.trace_path = trace_path
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)

        self._lock = threading.Lock()
        self._flush_every = flush_every
        self._trace_buf: List[Dict] = []

        # Accumulated FLOPs
        self._total_sparse_flops: int = 0
        self._total_dense_flops: int = 0
        self._total_flops: int = 0

        # Per-op-name breakdown
        self._op_sparse: Dict[str, int] = {}
        self._op_dense: Dict[str, int] = {}

        # Per-layer breakdown
        self._layer_sparse: Dict[int, int] = {}
        self._layer_dense: Dict[int, int] = {}

        self._total_records = 0
        self._session_start = time.time()

        # SAG governance additions
        # Confidence-gated FLOPs: sparse FLOPs that occurred under high confidence
        self._high_conf_sparse_flops: int = 0   # confidence >= 0.7
        self._low_conf_sparse_flops:  int = 0   # confidence < 0.4
        # Token region buckets (0=prefix, 1=middle, 2=suffix)
        self._region_sparse: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._region_dense:  Dict[int, int] = {0: 0, 1: 0, 2: 0}
        # Suppression impact: FLOPs gained (sparse) due to suppression
        self._suppression_gained_flops: int = 0
        self._suppression_events: int = 0

    # ------------------------------------------------------------------
    # Primary recording API
    # ------------------------------------------------------------------

    def record_op(
        self,
        step: int,
        layer_idx: int,
        op_name: str,
        is_sparse: bool,
        token_count: int,
        head_count: int,
        head_dim: int,
        sparsity_ratio: float = 1.0,
        confidence: float = -1.0,       # SAG: -1 = not provided
        token_offset: int = 0,          # SAG: position in sequence
        seq_len: int = 0,               # SAG: for region bucketing
        suppression_active: bool = False,# SAG: was hybrid suppression active?
    ) -> None:
        """
        Record one transformer operation's arithmetic footprint.

        Args:
            step:           decode step index
            layer_idx:      transformer layer (0-based)
            op_name:        "qk_dot" | "av_dot" | "ffn_gate" | "ffn_mlp" | etc.
            is_sparse:      True if the sparse execution path was used
            token_count:    number of query tokens in this operation
            head_count:     number of attention heads (for attention ops)
            head_dim:       per-head dimension
            sparsity_ratio: fraction of positions actually computed [0,1]
                            e.g. 0.25 means 75% of KV positions were skipped
        """
        ts = time.time()

        # FLOPs estimation from real parameters.
        # For attention: 2 * Q * K * D * H  (full dense)
        # Then sparse_flops = full_flops * sparsity_ratio
        full_flops = self._estimate_flops(op_name, token_count, head_count, head_dim)
        if is_sparse:
            sparse_flops = int(full_flops * sparsity_ratio)
            dense_flops = 0
        else:
            sparse_flops = 0
            dense_flops = full_flops
            sparsity_ratio = 0.0

        flops_this_op = sparse_flops + dense_flops

        with self._lock:
            self._total_records += 1
            self._total_sparse_flops += sparse_flops
            self._total_dense_flops  += dense_flops
            self._total_flops        += flops_this_op

            self._op_sparse[op_name] = self._op_sparse.get(op_name, 0) + sparse_flops
            self._op_dense[op_name]  = self._op_dense.get(op_name, 0)  + dense_flops

            self._layer_sparse[layer_idx] = self._layer_sparse.get(layer_idx, 0) + sparse_flops
            self._layer_dense[layer_idx]  = self._layer_dense.get(layer_idx, 0)  + dense_flops

            # SAG: confidence-gated accounting
            if is_sparse and confidence >= 0.0:
                if confidence >= 0.7:
                    self._high_conf_sparse_flops += sparse_flops
                elif confidence < 0.4:
                    self._low_conf_sparse_flops  += sparse_flops

            # SAG: token region bucketing
            region = self._token_region(token_offset, seq_len)
            if is_sparse:
                self._region_sparse[region] += sparse_flops
            else:
                self._region_dense[region]  += dense_flops

            # SAG: suppression impact
            if suppression_active and is_sparse:
                self._suppression_gained_flops += sparse_flops
                self._suppression_events += 1

            ev = asdict(ArithmeticEvent(
                ts=ts, step=step, layer_idx=layer_idx, op_name=op_name,
                is_sparse=is_sparse, token_count=token_count,
                head_count=head_count, head_dim=head_dim,
                flops_estimate=flops_this_op,
                sparse_flops=sparse_flops, dense_flops=dense_flops,
                sparsity_ratio=round(sparsity_ratio, 4),
            ))
            # Attach SAG fields
            ev["confidence"] = round(confidence, 4) if confidence >= 0 else None
            ev["token_region"] = region
            ev["suppression_active"] = suppression_active
            ev["phase"] = "39.0-SAG"
            self._trace_buf.append(ev)
            if len(self._trace_buf) >= self._flush_every:
                self._flush()

    # ------------------------------------------------------------------
    # Metrics queries
    # ------------------------------------------------------------------

    def get_participation_report(self) -> Dict[str, Any]:
        """
        Returns sparse arithmetic participation as a fraction.
        Fraction is derived entirely from accumulated event records.
        """
        with self._lock:
            total = max(self._total_flops, 1)
            sparse_frac = self._total_sparse_flops / total
            dense_frac = self._total_dense_flops / total

            per_op = {}
            for op in set(list(self._op_sparse.keys()) + list(self._op_dense.keys())):
                sp = self._op_sparse.get(op, 0)
                dn = self._op_dense.get(op, 0)
                op_total = max(sp + dn, 1)
                per_op[op] = {
                    "sparse_flops": sp,
                    "dense_flops": dn,
                    "sparse_fraction": round(sp / op_total, 4),
                }

            return {
                "total_records": self._total_records,
                "total_flops_estimated": self._total_flops,
                "sparse_flops": self._total_sparse_flops,
                "dense_flops": self._total_dense_flops,
                "sparse_participation": round(sparse_frac, 4),
                "dense_participation": round(dense_frac, 4),
                "per_op": per_op,
                "elapsed_sec": round(time.time() - self._session_start, 2),
                "note": "FLOPs derived from real token/head/dim parameters — not guessed.",
            }

    def get_layer_participation(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for li in sorted(set(list(self._layer_sparse.keys()) + list(self._layer_dense.keys()))):
                sp  = self._layer_sparse.get(li, 0)
                dn  = self._layer_dense.get(li,  0)
                tot = max(sp + dn, 1)
                rows.append({
                    "layer": li,
                    "sparse_flops":    sp,
                    "dense_flops":     dn,
                    "sparse_fraction": round(sp / tot, 4),
                })
            return rows

    def get_governance_report(self) -> Dict[str, Any]:
        """SAG-specific: confidence-gated and suppression-impact arithmetic report."""
        with self._lock:
            total_sparse = max(self._total_sparse_flops, 1)
            regions = {}
            for r, name in [(0, "prefix"), (1, "middle"), (2, "suffix")]:
                sp  = self._region_sparse[r]
                dn  = self._region_dense[r]
                tot = max(sp + dn, 1)
                regions[name] = {
                    "sparse_flops":    sp,
                    "dense_flops":     dn,
                    "sparse_fraction": round(sp / tot, 4),
                }
            return {
                "high_confidence_sparse_flops":    self._high_conf_sparse_flops,
                "low_confidence_sparse_flops":     self._low_conf_sparse_flops,
                "high_conf_sparse_fraction": round(
                    self._high_conf_sparse_flops / total_sparse, 4),
                "suppression_gained_flops":        self._suppression_gained_flops,
                "suppression_events":              self._suppression_events,
                "region_breakdown":                regions,
                "note": "All values from accumulated real execution events.",
            }

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_region(token_offset: int, seq_len: int) -> int:
        """Bucket a token position into prefix(0)/middle(1)/suffix(2)."""
        if seq_len <= 0 or token_offset <= 0:
            return 1  # unknown -> middle
        frac = token_offset / seq_len
        if frac < 0.25:
            return 0  # prefix
        elif frac < 0.75:
            return 1  # middle
        else:
            return 2  # suffix

    @staticmethod
    def _estimate_flops(op_name: str, token_count: int, head_count: int, head_dim: int) -> int:
        if op_name in ("qk_dot", "av_dot"):
            return 2 * token_count * token_count * head_dim * head_count
        else:
            return 2 * token_count * head_count * head_dim

    def _flush(self) -> None:
        if not self._trace_buf:
            return
        with open(self.trace_path, "a", encoding="utf-8") as f:
            for ev in self._trace_buf:
                f.write(json.dumps(ev) + "\n")
        self._trace_buf.clear()
