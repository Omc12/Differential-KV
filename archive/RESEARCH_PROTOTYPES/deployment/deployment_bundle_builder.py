import os
import shutil
import tarfile
import json
import time
from typing import List, Dict, Any

class DeploymentBundleBuilder:
    """
    Builds deployable runtime bundles, packages dependencies, and generates artifacts.
    """
    def __init__(self, workspace_root: str, output_dir: str = "dist"):
        self.workspace_root = workspace_root
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def build_bundle(self, bundle_name: str, includes: List[str] = None) -> str:
        timestamp = int(time.time())
        bundle_id = f"{bundle_name}_{timestamp}"
        bundle_path = os.path.join(self.output_dir, bundle_id)
        
        if not os.path.exists(bundle_path):
            os.makedirs(bundle_path)

        # 1. Copy relevant source files
        source_includes = includes or ["runtime", "serving", "api", "deployment", "anchor_logic", "analysis"]
        for subdir in source_includes:
            src = os.path.join(self.workspace_root, subdir)
            if os.path.exists(src):
                shutil.copytree(src, os.path.join(bundle_path, subdir), dirs_exist_ok=True)

        # 2. Generate requirements.txt
        # For simplicity, we copy the existing one or generate a basic one
        req_src = os.path.join(self.workspace_root, "requirements.txt")
        if os.path.exists(req_src):
            shutil.copy(req_src, os.path.join(bundle_path, "requirements.txt"))
        else:
            with open(os.path.join(bundle_path, "requirements.txt"), "w") as f:
                f.write("torch\nfastapi\nuvicorn\npyyaml\npydantic\n")

        # 3. Create entrypoint script
        entrypoint_content = """#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python -m serving.openai_compatible_api_gateway
"""
        with open(os.path.join(bundle_path, "start_serving.sh"), "w") as f:
            f.write(entrypoint_content)
        os.chmod(os.path.join(bundle_path, "start_serving.sh"), 0o755)

        # 4. Generate bundle manifest
        manifest = {
            "bundle_id": bundle_id,
            "created_at": time.ctime(timestamp),
            "version": "1.0.0",
            "components": source_includes
        }
        with open(os.path.join(bundle_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)

        # 5. Compress into tarball
        tarball_path = f"{bundle_path}.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(bundle_path, arcname=bundle_id)
            
        print(f"[DBB] Deployment bundle created: {tarball_path}")
        return tarball_path

    def validate_bundle(self, tarball_path: str) -> bool:
        """
        Validates the consistency of a created bundle.
        """
        if not os.path.exists(tarball_path):
            return False
        
        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                members = tar.getnames()
                # Check for essential files
                essential = ["requirements.txt", "manifest.json", "start_serving.sh"]
                for e in essential:
                    if not any(m.endswith(e) for m in members):
                        print(f"[DBB] Validation failed: Missing {e}")
                        return False
            return True
        except Exception as e:
            print(f"[DBB] Validation error: {e}")
            return False
