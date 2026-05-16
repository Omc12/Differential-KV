import os
import shutil
import json
import tarfile
import time

class ReproducibilityBundleExporter:
    """
    MANDATORY PHASE 18D: Packages all artifacts required for external verification.
    """
    def __init__(self, run_id: str, results_dir: str = "results/reconstruction_18/"):
        self.run_id = run_id
        self.results_dir = results_dir
        self.bundle_dir = os.path.join(results_dir, f"bundle_{run_id}")
        os.makedirs(self.bundle_dir, exist_ok=True)

    def add_artifact(self, source_path: str):
        if os.path.exists(source_path):
            shutil.copy(source_path, self.bundle_dir)
            return True
        return False

    def export_bundle(self):
        bundle_name = f"diffkv_repro_bundle_{self.run_id}_{int(time.time())}.tar.gz"
        bundle_path = os.path.join(self.results_dir, bundle_name)
        
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(self.bundle_dir, arcname=os.path.basename(self.bundle_dir))
            
        print(f"[PHASE 18D] Reproducibility bundle exported to {bundle_path}")
        return bundle_path

if __name__ == "__main__":
    exporter = ReproducibilityBundleExporter("test_run")
    print("Bundle Exporter ready.")
