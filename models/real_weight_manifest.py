import os
import json

class RealWeightManifest:
    """
    PHASE 18.1A: Logs physical weight files and their metadata for audit.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.manifest_path = "results/reconstruction_18_1/raw_checkpoint_manifest.json"

    def record_weights(self, checkpoint_dir: str):
        if not os.path.exists(checkpoint_dir):
            return {"status": "MISSING"}
            
        files = os.listdir(checkpoint_dir)
        weight_files = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
        
        manifest = {
            "model_id": self.model_id,
            "weights": []
        }
        
        for f in weight_files:
            f_path = os.path.join(checkpoint_dir, f)
            manifest["weights"].append({
                "name": f,
                "size_gb": os.path.getsize(f_path) / (1024**3),
                "mtime": os.path.getmtime(f_path)
            })
            
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        return manifest
