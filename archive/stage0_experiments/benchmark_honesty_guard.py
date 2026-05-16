import json
import os
from datetime import datetime
from typing import Dict, Any, List
from benchmark_mode_classifier import BenchmarkMode
from benchmark_component_registry import registry
from telemetry_scope_tracker import scope_tracker

class BenchmarkHonestyGuard:
    """
    Generates explicit benchmark honesty reports and manifests.
    """
    def __init__(self, mode: BenchmarkMode):
        self.mode = mode
        self.timestamp = datetime.now().isoformat()

    def generate_report_markdown(self) -> str:
        manifest = registry.get_participation_manifest()
        excluded = registry.get_excluded_manifest()
        scope = scope_tracker.get_scope_manifest()

        report = f"""# BENCHMARK CLASSIFICATION

MODE:
{self.mode.name}

INCLUDED:
{", ".join(manifest) if manifest else "None"}

EXCLUDED:
{", ".join(excluded) if excluded else "None"}

REAL MODEL WEIGHTS:
{"YES" if scope["model_weights"] else "NO"}

REAL LOGITS:
{"YES" if "logits" in manifest else "NO"}

REAL SAMPLING:
{"YES" if "sampling" in manifest else "NO"}

REAL WALL CLOCK:
{"YES" if scope["wall_clock"] else "NO"}

REAL VRAM:
{"YES" if scope["gpu_allocations"] else "NO"}

SYNTHETIC ACCOUNTING:
{"YES" if not scope["wall_clock"] or not scope["gpu_allocations"] else "NO"}
"""
        return report

    def save_manifests(self):
        honesty_manifest = {
            "mode": self.mode.name,
            "timestamp": self.timestamp,
            "included": registry.get_participation_manifest(),
            "excluded": registry.get_excluded_manifest(),
            "scope": scope_tracker.get_scope_manifest()
        }
        
        with open("benchmark_honesty_manifest.json", "w") as f:
            json.dump(honesty_manifest, f, indent=4)
            
        with open("benchmark_scope_manifest.json", "w") as f:
            json.dump(scope_tracker.get_scope_manifest(), f, indent=4)

    def write_mode_report(self, filename: str):
        content = self.generate_report_markdown()
        with open(filename, "w") as f:
            f.write(content)
        print(f"[BIC] Honesty report saved to {filename}")
