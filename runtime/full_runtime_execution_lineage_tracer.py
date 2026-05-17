"""
SIP Phase 41.2: Full Runtime Execution Lineage Tracer.

Purpose: Track COMPLETE request execution lineage to prove which layers actually run.
"""
import time
from typing import Dict, Any, List

class FullRuntimeExecutionLineageTracer:
    def __init__(self):
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._completed_lineages: List[Dict[str, Any]] = []

    def request_started(self, request_id: str, entrypoint: str):
        self._active_requests[request_id] = {
            "request_id": request_id,
            "entrypoint": entrypoint,
            "nodes_visited": [],
            "start_time": time.time(),
            "has_scheduler_path": False,
            "has_sparse_routing": False,
            "has_governance": False,
            "has_native_scheduler": False,
            "has_dense_fallback": False,
            "has_repair": False,
            "has_webui_bridge": False,
            "has_streaming_layer": False,
        }
        self.mark_node(request_id, f"Entrypoint: {entrypoint}")

    def mark_node(self, request_id: str, node_name: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["nodes_visited"].append({
                "node": node_name,
                "timestamp": time.time()
            })

    def mark_scheduler_path(self, request_id: str, is_native: bool = False):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_scheduler_path"] = True
            if is_native:
                self._active_requests[request_id]["has_native_scheduler"] = True
            self.mark_node(request_id, "Scheduler")

    def mark_sparse_routing(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_sparse_routing"] = True
            self.mark_node(request_id, "SparseRouting")

    def mark_governance(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_governance"] = True
            self.mark_node(request_id, "Governance")

    def mark_dense_fallback(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_dense_fallback"] = True
            self.mark_node(request_id, "DenseFallback")

    def mark_repair(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_repair"] = True
            self.mark_node(request_id, "Repair")

    def mark_webui_bridge(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_webui_bridge"] = True
            self.mark_node(request_id, "WebUIBridge")

    def mark_streaming_layer(self, request_id: str):
        if request_id in self._active_requests:
            self._active_requests[request_id]["has_streaming_layer"] = True
            self.mark_node(request_id, "StreamingLayer")

    def request_completed(self, request_id: str) -> Dict[str, Any]:
        if request_id in self._active_requests:
            lineage = self._active_requests.pop(request_id)
            lineage["end_time"] = time.time()
            lineage["duration_sec"] = lineage["end_time"] - lineage["start_time"]
            lineage["is_complete_lineage"] = (
                lineage["has_scheduler_path"] and 
                lineage["has_sparse_routing"] and 
                lineage["has_governance"] and 
                lineage["has_streaming_layer"]
            )
            self._completed_lineages.append(lineage)
            return lineage
        return {}

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._completed_lineages)
        if total == 0:
            return {"total_lineages": 0, "complete_lineage_ratio": 0.0}
        complete = sum(1 for l in self._completed_lineages if l["is_complete_lineage"])
        return {
            "total_lineages": total,
            "complete_lineages": complete,
            "complete_lineage_ratio": complete / total,
            "native_scheduler_ratio": sum(1 for l in self._completed_lineages if l["has_native_scheduler"]) / total,
            "webui_bridge_ratio": sum(1 for l in self._completed_lineages if l["has_webui_bridge"]) / total
        }
