import json
import os

class FailureBoundaryExporter:
    """
    PHASE 18.1F: Exports failure boundaries (OOM, context collapse) to a structured manifest.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/failure_boundaries.json"):
        self.export_path = export_path
        self.failures = []

    def record_failure(self, context_len: int, users: int, error_msg: str):
        failure = {
            "context_len": context_len,
            "users": users,
            "error": error_msg,
            "status": "HARD_FAILURE"
        }
        self.failures.append(failure)
        self.export()

    def export(self):
        with open(self.export_path, 'w') as f:
            json.dump(self.failures, f, indent=4)
