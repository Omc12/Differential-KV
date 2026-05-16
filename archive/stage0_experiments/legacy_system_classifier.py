import os
import json
import logging
from typing import Dict, List, Any

class LegacySystemClassifier:
    """
    Classifies project files into various categories for refactoring and archival.
    Generates: platform_system_manifest.json
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.logger = logging.getLogger("LegacySystemClassifier")
        self.manifest_path = os.path.join(root_dir, "platform_system_manifest.json")
        
        self.categories = {
            "ACTIVE_PRODUCTION": [
                "differential_kv_cli.py", "requirements.txt", "pyproject.toml"
            ],
            "ACTIVE_RUNTIME": [
                "runtime/hf_diffkv_wrapper.py", "decode_pipeline_fusion_engine.py", 
                "runtime/kv_runtime_manager.py", "persistent_triton_dispatcher.py",
                "active_gpu_residency_controller.py"
            ],
            "ACTIVE_SERVING": [
                "serving/openai_compatible_api_gateway.py", "serving/sparse_request_scheduler.py",
                "serving/production_session_manager.py", "latency_aware_batch_controller.py"
            ],
            "ACTIVE_TELEMETRY": [
                "persistent_observability_layer.py", "operational_health_monitor.py",
                "real_streaming_stability_engine.py", "user_fairness_telemetry.py"
            ],
            "ACTIVE_DEPLOYMENT": [
                "deployment_reproducibility_manager.py", "runtime_recovery_controller.py",
                "memory_pressure_safety_system.py"
            ],
            "EXPERIMENTAL": [],
            "LEGACY": [],
            "ARCHIVED": [],
            "SUPERSEDED": [],
            "STAGE1_HISTORICAL": []
        }

    def classify_project(self) -> Dict[str, List[str]]:
        """
        Scans the root directory and classifies all files.
        """
        self.logger.info("Classifying project files...")
        manifest = {cat: [] for cat in self.categories.keys()}
        
        # Pre-populate with known active systems
        for cat, files in self.categories.items():
            manifest[cat].extend(files)

        all_files = []
        for root, dirs, files in os.walk(self.root_dir):
            if any(d in root for d in [".git", "__pycache__", "archive", "node_modules"]):
                continue
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), self.root_dir)
                all_files.append(rel_path)

        for f in all_files:
            # Check if already classified
            already_classified = False
            for cat in manifest:
                if f in manifest[cat]:
                    already_classified = True
                    break
            
            if already_classified:
                continue
                
            # Logic for classifying remaining files
            if f.startswith("build_phase") or f.startswith("setup_phase"):
                manifest["STAGE1_HISTORICAL"].append(f)
            elif f.startswith("run_phase") or f.startswith("run_reconstruction"):
                manifest["LEGACY"].append(f)
            elif f.startswith("run_") and "_validation" in f:
                # Keep recent validation scripts in ACTIVE_RUNTIME for now, 
                # or move to validation category
                if "run_pdm" in f or "run_xvm" in f or "run_lgs" in f:
                    manifest["ACTIVE_RUNTIME"].append(f)
                else:
                    manifest["SUPERSEDED"].append(f)
            elif "integrity_guard" in f:
                if "pdm" in f or "xvm" in f or "lgs" in f:
                    manifest["ACTIVE_PRODUCTION"].append(f)
                else:
                    manifest["SUPERSEDED"].append(f)
            elif f.endswith(".md"):
                if "audit" in f or "report" in f:
                    manifest["STAGE1_HISTORICAL"].append(f)
                else:
                    manifest["ACTIVE_PRODUCTION"].append(f)
            elif f.startswith("triton_"):
                manifest["ACTIVE_RUNTIME"].append(f)
            else:
                manifest["EXPERIMENTAL"].append(f)

        # Save manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        self.logger.info(f"Classification complete. Manifest saved to {self.manifest_path}")
        return manifest

    def get_manifest(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.manifest_path):
            return self.classify_project()
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
