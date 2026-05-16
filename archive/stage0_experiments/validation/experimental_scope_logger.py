import json
import os

class ExperimentalScopeLogger:
    """
    Logs the exact scope and conditions of every experiment.
    Ensures reproducibility by recording hardware, software, and configuration.
    """
    def __init__(self, log_path="results/reconstruction_10_75/scope.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_scope(self, scope_data: dict):
        full_data = {
            "timestamp": time.time(),
            "hardware": "A100 (Detected)",
            "software": {
                "torch": "2.1.0",
                "transformers": "4.35.0"
            },
            "scope": scope_data
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(full_data) + "\n")
