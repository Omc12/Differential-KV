import json
import os

class FailureBoundaryMapper:
    """Maps the boundaries where sparse reconstruction fails."""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.boundaries = {}

    def analyze_results(self, log_file):
        if not os.path.exists(log_file):
            return
            
        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    mode = data.get("mode")
                    ctx = data.get("ctx")
                    domain = data.get("domain", "default")
                    success = data.get("success")
                    
                    key = (mode, domain)
                    if key not in self.boundaries:
                        self.boundaries[key] = {"max_stable_ctx": 0, "failures": []}
                        
                    if success:
                        self.boundaries[key]["max_stable_ctx"] = max(self.boundaries[key]["max_stable_ctx"], ctx)
                    else:
                        self.boundaries[key]["failures"].append(ctx)
                except:
                    continue

    def export_report(self):
        report_path = os.path.join(self.results_dir, "failure_boundaries.json")
        serializable_boundaries = {str(k): v for k, v in self.boundaries.items()}
        with open(report_path, "w") as f:
            json.dump(serializable_boundaries, f, indent=4)
        print(f"Failure boundary report exported to {report_path}")
