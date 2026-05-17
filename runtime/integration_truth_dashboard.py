"""
SIP Phase 41.2: Integration Truth Dashboard.

Purpose: Provide LIVE visibility into active runtime layers, sparse systems,
native path activation, governance participation, and execution path integrity.
"""
import time
import threading
from typing import Optional

from runtime.full_runtime_execution_lineage_tracer import FullRuntimeExecutionLineageTracer
from runtime.stage_participation_verifier import StageParticipationVerifier
from runtime.webui_serving_path_auditor import WebUIServingPathAuditor
from runtime.native_path_activation_verifier import NativePathActivationVerifier
from runtime.sparse_participation_reality_meter import SparseParticipationRealityMeter

class IntegrationTruthDashboard:
    def __init__(
        self,
        lineage_tracer: FullRuntimeExecutionLineageTracer,
        stage_verifier: StageParticipationVerifier,
        path_auditor: WebUIServingPathAuditor,
        native_verifier: NativePathActivationVerifier,
        sparse_meter: SparseParticipationRealityMeter,
    ):
        self._lineage = lineage_tracer
        self._stage = stage_verifier
        self._path = path_auditor
        self._native = native_verifier
        self._sparse = sparse_meter
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sip_dashboard")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            try:
                self._print_snapshot()
            except Exception:
                pass
            time.sleep(2.0)

    def _print_snapshot(self):
        lineage_stats = self._lineage.get_stats()
        stage_stats = self._stage.get_stats()
        path_stats = self._path.get_audit_stats()
        native_stats = self._native.get_activation_stats()
        sparse_stats = self._sparse.get_participation_stats()

        active_layers = stage_stats.get('active_components', 0)
        total_layers = stage_stats.get('total_components', 1)
        layer_activation_pct = (active_layers / total_layers) * 100 if total_layers else 0.0

        sparse_pct = sparse_stats.get('sparse_participation_ratio', 0.0) * 100
        dense_pct = sparse_stats.get('dense_ratio', 0.0) * 100
        
        gov_activation_pct = lineage_stats.get('complete_lineage_ratio', 0.0) * 100
        native_pct = native_stats.get('native_execution_ratio', 0.0) * 100
        
        path_integrity = path_stats.get('path_integrity_score', 0.0) * 100
        lineage_completeness = lineage_stats.get('complete_lineage_ratio', 0.0) * 100

        print(
            f"\n{'='*75}\n"
            f"[SIP INTEGRATION TRUTH]  {time.strftime('%H:%M:%S')}\n"
            f"{'='*75}\n"
            f"  Active Runtime Layers     : {active_layers}/{total_layers} ({layer_activation_pct:.1f}%)\n"
            f"  Sparse Participation      : {sparse_pct:.1f}%\n"
            f"  Governance Activation     : {gov_activation_pct:.1f}%\n"
            f"  Native Execution          : {native_pct:.1f}%\n"
            f"  Dense Fallback/Bypass     : {dense_pct:.1f}% (Bypass Events: {path_stats.get('total_bypasses', 0)})\n"
            f"  WebUI Path Integrity      : {path_integrity:.1f}%\n"
            f"  Execution Lineage Complete: {lineage_completeness:.1f}%\n"
            f"{'='*75}"
        )
