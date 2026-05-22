"""
RCO-N C++ Native Modules Compiler.
Compiles all C++ pybind11 hot-path components with CMake for Windows compatibility.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

def build_module(name: str, src_dir: Path):
    print(f"\n========================================================")
    print(f"Building Native Module: {name}")
    print(f"========================================================")
    
    build_dir = src_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run CMake Configure
        print(f"Configuring {name} with CMake...")
        res = subprocess.run(
            ["cmake", "..", "-G", "Visual Studio 17 2022", "-A", "x64"],
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode != 0:
            print("CMake Configuration failed! Trying default generator...")
            res = subprocess.run(
                ["cmake", ".."],
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if res.returncode != 0:
                print(res.stderr)
                return False

        # Run CMake Build
        print(f"Building {name} with Release configuration...")
        res_build = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res_build.returncode != 0:
            print("Build failed!")
            print(res_build.stderr)
            return False
        
        # Copy compiled extension up to the source directory
        print("Locating compiled binary...")
        release_dir = build_dir / "Release"
        pyd_files = list(release_dir.glob("*.pyd"))
        if not pyd_files:
            # Maybe it compiled in the main build folder
            pyd_files = list(build_dir.glob("*.pyd"))
            
        if pyd_files:
            target = src_dir / pyd_files[0].name
            shutil.copy2(pyd_files[0], target)
            print(f"[SUCCESS] Native C++ module {name} compiled & copied to: {target}")
            return True
        else:
            print("Could not find compiled .pyd file!")
            return False
            
    except Exception as e:
        print(f"Error compiling {name}: {e}")
        return False

def main():
    root = Path("d:/Codes/Projects/Differential KV/native")
    modules = [
        ("native_decode_scheduler", root / "native_decode_scheduler"),
        ("native_sparse_metadata_engine", root / "native_sparse_metadata_engine"),
        ("native_telemetry_counter_layer", root / "native_telemetry_counter_layer"),
        ("native_sparse_cuda_extension", root / "native_sparse_cuda_extension"),
    ]
    
    success_count = 0
    for name, path in modules:
        if build_module(name, path):
            success_count += 1
            
    print(f"\n========================================================")
    print(f"Compilation finished. Compiled {success_count}/{len(modules)} native modules.")
    print(f"========================================================")

if __name__ == "__main__":
    main()
