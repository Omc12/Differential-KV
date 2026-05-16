import json

class CrossRunConsistencyChecker:
    """Verifies consistency of metrics and outputs across repeated runs."""
    
    def __init__(self):
        self.run_history = {}

    def add_run(self, run_id, result):
        key = (result["mode"], result["ctx"], result.get("domain", "default"))
        if key not in self.run_history:
            self.run_history[key] = []
        self.run_history[key].append(result)

    def check_consistency(self):
        report = {}
        for key, runs in self.run_history.items():
            if len(runs) < 2:
                continue
                
            successes = [r["success"] for r in runs]
            tps_values = [r["tps"] for r in runs]
            vram_values = [r["vram_gb"] for r in runs]
            
            report[str(key)] = {
                "success_consistency": all(s == successes[0] for s in successes),
                "tps_variance": max(tps_values) - min(tps_values),
                "vram_variance": max(vram_values) - min(vram_values),
                "num_runs": len(runs)
            }
        return report
