import os
import json
from .config_fingerprint import ConfigFingerprint
from .environment_snapshot import EnvironmentSnapshot

class ReproducibilityRunner:
    """
    Ensures that benchmark runs are fully reproducible.
    Captures config fingerprints and hardware environment snapshots.
    """
    def __init__(self, output_dir="results/reconstruction_10/reproducibility"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.fingerprinter = ConfigFingerprint()
        self.snapshotter = EnvironmentSnapshot()

    def record_run(self, run_id, config):
        print(f"[Reproducibility] Recording run metadata for: {run_id}")
        
        fingerprint = self.fingerprinter.generate(config)
        env = self.snapshotter.capture()
        
        metadata = {
            "run_id": run_id,
            "config_fingerprint": fingerprint,
            "environment": env,
            "timestamp": os.times()[4]
        }
        
        path = os.path.join(self.output_dir, f"run_{run_id}_meta.json")
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return path
