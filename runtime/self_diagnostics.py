"""
runtime/self_diagnostics.py

Continuous monitoring and telemetry for the Unified Cognitive Runtime.
Generates heatmaps, intervention logs, and manifold health traces.
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict

class SelfDiagnostics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        self.intervention_logs: List[Dict[str, Any]] = []
        self.telemetry_history: List[Dict[str, Any]] = []
        self.manifold_traces: List[List[float]] = [] # [step, stability, drift]

    def log_intervention(self, step: int, intervention_type: str, details: Dict[str, Any]):
        self.intervention_logs.append({
            "step": step,
            "timestamp": time.time() - self.start_time,
            "type": intervention_type,
            "details": details
        })

    def update_telemetry(self, step: int, health_state: Any, vram_usage: int):
        record = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            "health": health_state.cognitive_health_score,
            "collapse_risk": health_state.collapse_probability,
            "vram_bytes": vram_usage,
            "drift": health_state.latent_drift
        }
        self.telemetry_history.append(record)
        self.manifold_traces.append([step, health_state.manifold_stability, health_state.latent_drift])

    def generate_report(self) -> Dict[str, Any]:
        duration = time.time() - self.start_time
        return {
            "runtime_duration": duration,
            "total_interventions": len(self.intervention_logs),
            "avg_health": sum(t["health"] for t in self.telemetry_history) / len(self.telemetry_history) if self.telemetry_history else 1.0,
            "peak_collapse_risk": max((t["collapse_risk"] for t in self.telemetry_history), default=0.0),
            "vram_efficiency": "High" if self.telemetry_history and self.telemetry_history[-1]["vram_bytes"] < 4e9 else "Moderate",
            "intervention_summary": self._summarize_interventions()
        }

    def _summarize_interventions(self) -> Dict[str, int]:
        summary = {}
        for log in self.intervention_logs:
            t = log["type"]
            summary[t] = summary.get(t, 0) + 1
        return summary

    def save_telemetry(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "telemetry": self.telemetry_history,
                "interventions": self.intervention_logs,
                "traces": self.manifold_traces
            }, f, indent=2)
