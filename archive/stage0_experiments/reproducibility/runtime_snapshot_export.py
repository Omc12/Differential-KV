"""
reproducibility/runtime_snapshot_export.py

Exports Differential KV runtime configurations and model weights for independent verification.
Ensures full transparency of deployment states.
"""

import torch
import json
import os
import shutil
from typing import Dict, Any

class RuntimeSnapshotExporter:
    def __init__(self, export_dir: str = "results/phase38/snapshots"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_config(self, config: Dict[str, Any], name: str = "runtime_config.json"):
        path = os.path.join(self.export_dir, name)
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Config exported to {path}")

    def export_model_state(self, model, name: str = "model_state.pt"):
        path = os.path.join(self.export_dir, name)
        # Only export patched weights or anchors to save space
        state_dict = {k: v for k, v in model.state_dict().items() if "ncaa" in k or "anchor" in k}
        torch.save(state_dict, path)
        print(f"Model state exported to {path}")

    def bundle_all(self, bundle_name: str = "validation_bundle.zip"):
        shutil.make_archive(
            os.path.join(self.export_dir, "..", bundle_name.replace(".zip", "")),
            'zip',
            self.export_dir
        )
        print(f"Bundle created: {bundle_name}")

if __name__ == "__main__":
    exporter = RuntimeSnapshotExporter()
    exporter.export_config({"mode": "differential", "version": "v1.0"})
    print("Snapshot export demonstration completed.")
