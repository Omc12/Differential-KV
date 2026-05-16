import time
from typing import Dict, List, Any

class RuntimeDensityProfiler:
    """
    Identifies which components remain dense during execution.
    Generates component-level runtime dominance reports.
    """
    def __init__(self):
        self.timers = {}
        self.density_map = {
            "attention": "sparse",
            "mlp": "dense",
            "projections": "dense",
            "sampling": "dense",
            "logits": "dense",
            "kv_movement": "sparse"
        }

    def start(self, component: str):
        if component not in self.timers:
            self.timers[component] = {"total_time": 0, "calls": 0}
        self.timers[component]["start"] = time.perf_counter()

    def end(self, component: str):
        if component in self.timers and "start" in self.timers[component]:
            duration = time.perf_counter() - self.timers[component]["start"]
            self.timers[component]["total_time"] += duration
            self.timers[component]["calls"] += 1

    def get_report(self) -> Dict[str, Any]:
        total_runtime = sum(t["total_time"] for t in self.timers.values())
        if total_runtime == 0:
            return {}

        report = []
        sparse_runtime = 0
        dense_runtime = 0

        for comp, stats in self.timers.items():
            percent = (stats["total_time"] / total_runtime) * 100
            mode = self.density_map.get(comp, "unknown")
            if mode == "sparse":
                sparse_runtime += stats["total_time"]
            else:
                dense_runtime += stats["total_time"]
            
            report.append({
                "component": comp,
                "runtime_percent": percent,
                "mode": mode
            })

        dominant_dense = max(
            [r for r in report if r["mode"] == "dense"], 
            key=lambda x: x["runtime_percent"],
            default={"component": "none"}
        )["component"]

        return {
            "components": report,
            "dense_runtime_percent": (dense_runtime / total_runtime) * 100,
            "sparse_runtime_percent": (sparse_runtime / total_runtime) * 100,
            "dominant_dense_component": dominant_dense
        }

    def print_dominance_report(self):
        data = self.get_report()
        print("\n" + "="*40)
        print("RUNTIME DENSITY DOMINANCE REPORT")
        print("="*40)
        print(f"Sparse Runtime: {data['sparse_runtime_percent']:.2f}%")
        print(f"Dense Runtime:  {data['dense_runtime_percent']:.2f}%")
        print(f"Dominant Dense: {data['dominant_dense_component']}")
        print("-" * 40)
        for comp in data["components"]:
            print(f"{comp['component']:15} | {comp['runtime_percent']:5.1f}% | {comp['mode']}")
        print("="*40 + "\n")

# Global singleton
profiler = RuntimeDensityProfiler()
