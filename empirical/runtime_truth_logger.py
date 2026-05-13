import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

class RuntimeTruthLogger:
    """
    Strict empirical logger that saves raw telemetry without projections.
    """
    def __init__(self, run_id: str, base_dir: str = "results/reconstruction_6_5"):
        self.run_id = run_id
        self.log_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "raw_telemetry.json")
        self.entries = []

    def log(self, category: str, data: Dict[str, Any]):
        """Logs a single empirical measurement."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "category": category,
            **data
        }
        self.entries.append(entry)
        
        # Incremental save to prevent data loss in long runs
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_summary(self, summary: Dict[str, Any]):
        """Saves a final run summary."""
        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

    def get_log_path(self) -> str:
        return self.log_file

if __name__ == "__main__":
    logger = RuntimeTruthLogger("test_run")
    logger.log("tps", {"value": 150.5, "unit": "tokens/s"})
    print(f"Logged to {logger.get_log_path()}")
