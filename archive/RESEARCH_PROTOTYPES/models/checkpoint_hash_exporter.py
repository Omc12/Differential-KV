import hashlib
import os
import json

class CheckpointHashExporter:
    """
    PHASE 18.1A: Exports final hashes for model, tokenizer, and config.
    MANDATORY: MUST export before benchmarking.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/checkpoint_hashes.json"):
        self.export_path = export_path
        os.makedirs(os.path.dirname(self.export_path), exist_ok=True)

    def export(self, manifest_data: dict):
        print(f"[PHASE 18.1A] Exporting Checkpoint Hashes to {self.export_path}")
        
        with open(self.export_path, 'w') as f:
            json.dump(manifest_data, f, indent=4)
            
        return self.export_path
