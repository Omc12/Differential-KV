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
                "differential_kv_cli.py", "requirements.txt", "pyproject.toml",
                "STAGE1_FINAL_ARCHITECTURE.md", "RUNTIME_FLOW_MAP.md",
                "SPARSE_RUNTIME_OVERVIEW.md", "DEPLOYMENT_GUIDE.md",
                "BENCHMARKING_GUIDE.md"
            ],
            "ACTIVE_RUNTIME": [
                "runtime/hf_dkv_wrapper.py", "decode_pipeline_fusion_engine.py", 
                "runtime/kv_runtime_manager.py", "persistent_triton_dispatcher.py",
                "active_gpu_residency_controller.py", "triton_token_collapse_kernel.py",
                "triton_sparse_mlp_kernel.py", "occupancy_aware_triton_fuser.py",
                "runtime_activation_controller.py", "capability_runtime_detector.py"
            ],
            "ACTIVE_SERVING": [
                "serving/openai_compatible_api_gateway.py", "serving/sparse_request_scheduler.py",
                "serving/production_session_manager.py", "latency_aware_batch_controller.py",
                "serving_overhead_minimizer.py", "sparse_qos_stabilizer.py"
            ],
            "ACTIVE_TELEMETRY": [
                "persistent_observability_layer.py", "operational_health_monitor.py",
                "real_streaming_stability_engine.py", "user_fairness_telemetry.py",
                "production_serving_telemetry.py", "real_hardware_sparse_telemetry.py"
            ],
            "ACTIVE_DEPLOYMENT": [
                "deployment_reproducibility_manager.py", "runtime_recovery_controller.py",
                "memory_pressure_safety_system.py", "quick_start.py"
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
                rel_path = rel_path.replace(os.sep, '/')
                all_files.append(rel_path)

        for f in all_files:
            # PROTECT PRC SCRIPTS
            if "prc_" in f or "run_prc" in f or "legacy_system_classifier" in f or "historical_archive_manager" in f:
                manifest["ACTIVE_PRODUCTION"].append(f)
                continue

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
            elif f.startswith("run_phase") or f.startswith("run_reconstruction") or f.startswith("run_ako") or f.startswith("run_bso") or f.startswith("run_dco") or f.startswith("run_cko") or f.startswith("run_dko"):
                manifest["LEGACY"].append(f)
            elif f.startswith("run_") and "_validation" in f:
                if any(x in f for x in ["run_pdm", "run_xvm", "run_lgs", "run_atc", "run_eom", "run_hsm"]):
                    manifest["ACTIVE_RUNTIME"].append(f)
                else:
                    manifest["SUPERSEDED"].append(f)
            elif "integrity_guard" in f:
                if any(x in f for x in ["pdm", "xvm", "lgs", "atc", "eom", "hsm", "prc"]):
                    manifest["ACTIVE_PRODUCTION"].append(f)
                else:
                    manifest["SUPERSEDED"].append(f)
            elif f.endswith(".md"):
                if any(x in f for x in ["audit", "report", "appendix", "summary"]):
                    manifest["STAGE1_HISTORICAL"].append(f)
                else:
                    manifest["ACTIVE_PRODUCTION"].append(f)
            elif f.startswith("runtime/") or f.startswith("serving/") or f.startswith("integrations/") or f.startswith("telemetry/") or f.startswith("compression/") or f.startswith("models/") or f.startswith("kernels/") or f.startswith("virtualization/") or f.startswith("optimization/") or f.startswith("execution/") or f.startswith("orchestration/") or f.startswith("safety/") or f.startswith("reproducibility/"):
                # These are active directories
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
