import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path

class NativeAsyncStreamCoordinator:
    """
    NDX Phase 42.1.5 — Native Async Stream Coordinator.
    Coordinates CUDA streams natively.
    No silent Python timing coordination allowed.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        
        # Clean up any previously compiled dynamic DLLs
        for p in self.workspace_root.glob("runtime/native_async_stream_coordinator_*.dll"):
            try:
                os.remove(p)
            except Exception:
                pass
                
        self.lib_path = self.workspace_root / f"runtime/native_async_stream_coordinator_{int(time.time())}.dll"
        self.dll = None
        self.compiled = False
        
        self._compile_and_load()

    def _compile_and_load(self):
        cpp_file = self.workspace_root / "runtime/native_async_stream_coordinator.cpp"
        
        mingw_bin = "C:\\ProgramData\\mingw64\\mingw64\\bin"
        if not os.environ.get("PATH", "").startswith(mingw_bin):
            os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(mingw_bin)
            except Exception:
                pass
                
        env = os.environ.copy()
        
        if cpp_file.exists():
            try:
                print("[Native Compile] Compiling native stream coordinator DLL...")
                if self.lib_path.exists():
                    try:
                        os.remove(self.lib_path)
                    except Exception:
                        pass
                
                res = subprocess.run(
                    [
                        "C:\\ProgramData\\mingw64\\mingw64\\bin\\g++.exe", "-O3", "-shared", "-std=c++11",
                        str(cpp_file),
                        "-o", str(self.lib_path)
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                if res.returncode == 0:
                    print(f"[Native Compile] Stream Coordinator DLL compiled successfully: {self.lib_path}")
            except Exception as e:
                print(f"[Native Compile] Warning: Could not run g++: {e}")
                
        if self.lib_path.exists():
            try:
                self.dll = ctypes.CDLL(str(self.lib_path))
                
                self.dll.init_stream_coordinator.argtypes = []
                self.dll.init_stream_coordinator.restype = None
                
                self.dll.native_trigger_overlap.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong]
                self.dll.native_trigger_overlap.restype = None
                
                self.dll.get_last_overlap_ms.argtypes = []
                self.dll.get_last_overlap_ms.restype = ctypes.c_float
                
                self.dll.is_overlap_active.argtypes = []
                self.dll.is_overlap_active.restype = ctypes.c_int
                
                self.dll.init_stream_coordinator()
                self.compiled = True
                print("[Native Load] Stream Coordinator DLL successfully loaded via ctypes.")
            except Exception as e:
                print(f"[Native Load] Warning: Load Stream Coordinator DLL failed: {e}")
                self.compiled = False
        else:
            self.compiled = False

    def trigger_overlap(self, compute_stream, transfer_stream):
        if self.dll and self.compiled:
            # Safely extract raw stream pointers or simulate if none
            comp_ptr = getattr(compute_stream, "cuda_stream", 0)
            trans_ptr = getattr(transfer_stream, "cuda_stream", 0)
            self.dll.native_trigger_overlap(ctypes.c_ulonglong(comp_ptr), ctypes.c_ulonglong(trans_ptr))
            return
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in trigger_overlap!")

    def get_overlap_metrics(self) -> float:
        if self.dll and self.compiled:
            return float(self.dll.get_last_overlap_ms())
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in get_overlap_metrics!")

    def is_active(self) -> bool:
        if self.dll and self.compiled:
            return self.dll.is_overlap_active() == 1
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in is_active!")
