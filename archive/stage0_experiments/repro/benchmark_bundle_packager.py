import os
import tarfile
import time

class BenchmarkBundlePackager:
    """
    Phase 18F: Packages results, config, and environment into a single verifiable bundle.
    """
    def __init__(self, run_id: str, results_dir: str = "results/reconstruction_18/"):
        self.run_id = run_id
        self.results_dir = results_dir

    def package(self):
        bundle_path = os.path.join(self.results_dir, f"diffkv_scientific_bundle_{self.run_id}.tar.gz")
        
        with tarfile.open(bundle_path, "w:gz") as tar:
            # Add all files in the results dir
            for root, dirs, files in os.walk(self.results_dir):
                for file in files:
                    if not file.endswith(".tar.gz"):
                        file_path = os.path.join(root, file)
                        tar.add(file_path, arcname=os.path.relpath(file_path, self.results_dir))
                        
        print(f"[PHASE 18F] Scientific bundle packaged: {bundle_path}")
        return bundle_path
