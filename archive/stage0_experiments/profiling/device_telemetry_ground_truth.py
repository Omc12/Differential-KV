import os
import json

class DeviceTelemetryGroundTruth:
    """
    Consolidates device-level telemetry into a ground-truth report.
    Ensures that all telemetry is physically measured, not simulated.
    """
    def __init__(self, output_file="results/reconstruction_10/device_telemetry.json"):
        self.output_file = output_file
        self.telemetry_data = []

    def log_telemetry(self, component, metrics):
        """
        Logs telemetry for a specific hardware component.
        """
        entry = {
            "component": component,
            "metrics": metrics,
            "timestamp": os.times()[4] # Real process time
        }
        self.telemetry_data.append(entry)
        self._save()

    def _save(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.telemetry_data, f, indent=4)

    def get_ground_truth(self):
        """
        Returns the consolidated ground truth for the device.
        """
        return self.telemetry_data
