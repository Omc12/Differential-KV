import json
import os

class TelemetryReplayArchive:
    """
    Phase 18F: Stores raw telemetry events for deterministic replay/analysis.
    """
    def __init__(self, run_id: str, export_dir: str = "results/reconstruction_18/"):
        self.run_id = run_id
        self.export_dir = export_dir
        self.events = []

    def record_event(self, event_type, data):
        self.events.append({
            "type": event_type,
            "data": data
        })

    def archive(self):
        filename = f"telemetry_replay_{self.run_id}.jsonl"
        path = os.path.join(self.export_dir, filename)
        with open(path, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        return path
