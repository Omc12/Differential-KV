import json
import os

class RuntimeConfigSnapshot:
    """
    Captures the exact state of the system before a benchmark run.
    """
    def __init__(self, run_id: str, export_dir: str = "results/reconstruction_18/"):
        self.run_id = run_id
        self.export_dir = export_dir

    def snapshot(self, config_dict):
        filename = f"config_snapshot_{self.run_id}.json"
        path = os.path.join(self.export_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
        return path
