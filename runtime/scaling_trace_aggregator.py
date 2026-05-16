import json
import logging
from pathlib import Path
from typing import Dict, Any, List

class ScalingTraceAggregator:
    """
    SGC Phase 39.1: Scaling Trace Aggregator.
    Aggregates RAW metrics from isolated model run folders.
    NO conclusions. NO smoothing.
    """
    def __init__(self):
        self.logger = logging.getLogger("SGC_Aggregator")

    def aggregate_model_run(self, run_mgr: Any) -> Dict[str, Any]:
        """
        Reads raw trace files from a specific run_mgr and returns 
        the raw aggregated scalar metrics.
        """
        metrics = {
            "model_id": run_mgr.model_id,
            "participation_rate": 0.0,
            "mean_confidence": 0.0,
            "escalation_count": 0,
            "prevented_fallbacks": 0,
            "stable_layer_count": 0
        }

        # 1. Parse Arithmetic participation
        arith_path = Path(run_mgr.trace_path("arithmetic_governance_trace.jsonl"))
        if arith_path.exists():
            metrics["participation_rate"] = self._get_last_participation(arith_path)

        # 2. Parse Confidence
        conf_path = Path(run_mgr.trace_path("sparse_confidence_trace.jsonl"))
        if conf_path.exists():
            metrics["mean_confidence"] = self._get_avg_metric(conf_path, "confidence")

        # 3. Parse Escalations/Suppression
        supp_path = Path(run_mgr.trace_path("hybrid_suppression_audit.jsonl"))
        if supp_path.exists():
            supp_data = self._get_summary_from_suppression(supp_path)
            metrics["prevented_fallbacks"] = supp_data.get("prevented", 0)

        # 4. Parse Model ID from manifest (already in run_mgr, but verifying file)
        manifest_path = Path(run_mgr.manifest_path("manifest.json"))
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                metrics["model_id"] = manifest.get("model_id", run_mgr.model_id)

        return metrics

    def _get_last_participation(self, path: Path) -> float:
        total_sparse = 0.0
        total_dense = 0.0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_sparse += data.get("sparse_flops", 0)
                    total_dense += data.get("dense_flops", 0)
                except: continue
        total = total_sparse + total_dense
        return total_sparse / total if total > 0 else 0.0

    def _get_avg_metric(self, path: Path, key: str) -> float:
        vals = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if key in data and data[key] is not None: 
                        vals.append(data[key])
                except: continue
        return sum(vals) / len(vals) if vals else 0.0

    def _get_summary_from_suppression(self, path: Path) -> Dict[str, int]:
        prevented = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("suppressed") is True:
                        prevented += 1
                except: continue
        return {"prevented": prevented}

class SparseSurvivabilityCurveBuilder:
    """
    SGC Phase 39.1: Sparse Survivability Curve Builder.
    Constructs the JSON curve mapping model size to governance behavior.
    """
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.curves = {
            "model_sizes_b": [],
            "participation_curve": [],
            "confidence_curve": [],
            "escalation_curve": []
        }

    def add_point(self, size_b: float, metrics: Dict[str, Any]):
        self.curves["model_sizes_b"].append(size_b)
        self.curves["participation_curve"].append(metrics["participation_rate"])
        self.curves["confidence_curve"].append(metrics["mean_confidence"])
        self.curves["escalation_curve"].append(metrics["prevented_fallbacks"])

    def persist(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.curves, f, indent=4)
