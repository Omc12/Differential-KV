"""
benchmark_artifact_normalizer.py

Normalizes telemetry and metrics into a standardized CBP format.
Eliminates metric ambiguity across different measurement sources.
"""

from typing import Dict, Any, List
import numpy as np

class BenchmarkArtifactNormalizer:
    """
    Standardizes raw telemetry into canonical publication metrics.
    """
    
    def normalize_latency(self, latencies: List[float]) -> Dict[str, float]:
        """Calculates standardized percentiles."""
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "variance": 0.0}
        
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "variance": float(np.var(latencies)),
            "std_dev": float(np.std(latencies))
        }

    def normalize_metrics(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms raw telemetry into a flat, canonical metric set.
        """
        normalized = {}
        
        # Latency normalization
        latencies = raw_telemetry.get("latencies", [])
        latency_stats = self.normalize_latency(latencies)
        normalized["p50_latency_ms"] = latency_stats["p50"]
        normalized["p95_latency_ms"] = latency_stats["p95"]
        normalized["p99_latency_ms"] = latency_stats["p99"]
        normalized["latency_variance"] = latency_stats["variance"]
        
        # Throughput normalization
        tokens = raw_telemetry.get("total_tokens", 0)
        duration = raw_telemetry.get("total_duration_sec", 1.0)
        normalized["sustained_tps"] = tokens / max(0.001, duration)
        
        # TTFT and ITL
        normalized["ttft_ms"] = raw_telemetry.get("ttft_ms", 0.0)
        normalized["itl_ms"] = raw_telemetry.get("itl_ms", 0.0)
        
        # Resource Accounting
        normalized["vram_residency_mb"] = raw_telemetry.get("vram_usage_mb", 0.0)
        normalized["kv_residency_mb"] = raw_telemetry.get("kv_cache_usage_mb", 0.0)
        normalized["occupancy_stability"] = raw_telemetry.get("occupancy_rate", 1.0)
        
        # Overhead Accounting
        normalized["launch_overhead_ratio"] = raw_telemetry.get("launch_overhead", 0.0)
        normalized["serving_overhead_ratio"] = raw_telemetry.get("serving_overhead", 0.0)
        
        # Runtime participation
        normalized["sparse_runtime_pct"] = raw_telemetry.get("sparse_path_ratio", 1.0) * 100
        normalized["dense_runtime_pct"] = (1.0 - raw_telemetry.get("sparse_path_ratio", 1.0)) * 100
        
        return normalized

# Global instance
artifact_normalizer = BenchmarkArtifactNormalizer()
