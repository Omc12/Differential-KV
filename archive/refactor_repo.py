import os
import shutil
from pathlib import Path

def get_category(path_str):
    name = os.path.basename(path_str)
    
    # Active Runtime
    active_files = [
        "launch_real_serving.py", 
        "openai_compatible_api_gateway.py", 
        "production_session_manager.py",
        "lgs_resolver.py",
        "hf_diffkv_wrapper.py",
        "kv_runtime_manager.py",
        "requirements.txt",
        "README.md",
        "triton_diffkv.py"
    ]
    if name in active_files:
        return "ACTIVE_RUNTIME"
        
    # Experimental Runtime
    experimental_keywords = [
        "kernel", "triton", "cuda", "fused", "compression", "lowrank", "quantization", "sparse_repair", "attention"
    ]
    if any(k in name for k in experimental_keywords) and not name.endswith(".dll") and not name.endswith(".lib") and not name.endswith(".exp"):
        if not any(f in name for f in ["validator", "auditor", "guard", "telemetry", "trace", "benchmark"]):
            return "EXPERIMENTAL_RUNTIME"
            
    # Archived Synthetic
    archived_keywords = [
        "validator", "auditor", "guard", "telemetry", "trace", "benchmark", "simulator", 
        "report", "manifest", "verify", "run_phase", "run_", "generate_", "dashboard", "profiler",
        "stress", "harness"
    ]
    if any(k in name for k in archived_keywords):
        return "ARCHIVED_SYNTHETIC_SYSTEMS"
        
    if name.endswith(".dll") or name.endswith(".lib") or name.endswith(".exp") or name.endswith(".png") or name.endswith(".log"):
        return "ARCHIVED_SYNTHETIC_SYSTEMS"
        
    if "archive" in path_str.lower() or "traces" in path_str.lower():
        return "ARCHIVED_SYNTHETIC_SYSTEMS"
        
    return "RESEARCH_PROTOTYPES"

def main():
    base_dir = Path(r"d:\Codes\Projects\Differential KV")
    
    layers = ["ACTIVE_RUNTIME", "EXPERIMENTAL_RUNTIME", "RESEARCH_PROTOTYPES", "ARCHIVED_SYNTHETIC_SYSTEMS"]
    for l in layers:
        (base_dir / l).mkdir(exist_ok=True)
        
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in [".git", "__pycache__"] + layers]
        
        for file in files:
            if file == "refactor_repo.py" or file == "refactor_repo.py": continue
            
            src = Path(root) / file
            rel_path = src.relative_to(base_dir)
            
            category = get_category(str(rel_path))
            
            # Specific overrides for active API/serving if they exist
            if "api" in str(rel_path) or "serving" in str(rel_path):
                if category == "RESEARCH_PROTOTYPES":
                    pass
                    
            dest = base_dir / category / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.move(str(src), str(dest))
            except PermissionError:
                # If we can't move it because it's locked, just try to copy it instead and leave the original
                try:
                    shutil.copy2(str(src), str(dest))
                except Exception as e:
                    print(f"Failed to copy {src}: {e}")
            except Exception as e:
                print(f"Failed to move {src}: {e}")
            
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for d in dirs:
            dir_path = Path(root) / d
            if dir_path.name not in [".git"] + layers:
                try:
                    dir_path.rmdir()
                except OSError:
                    pass

if __name__ == "__main__":
    main()
