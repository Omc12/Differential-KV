import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path

class NativeQueueCollapseRuntime:
    """
    NDX Phase 42.1.5 — Native Queue Collapse Runtime.
    Eliminates Python queue turbulence entirely natively.
    No silent Python scheduling fallbacks allowed.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        
        # Clean up any previously compiled dynamic DLLs
        for p in self.workspace_root.glob("runtime/native_queue_collapse_runtime_*.dll"):
            try:
                os.remove(p)
            except Exception:
                pass
                
        self.lib_path = self.workspace_root / f"runtime/native_queue_collapse_runtime_{int(time.time())}.dll"
        self.dll = None
        self.compiled = False
        
        self._compile_and_load()

    def _compile_and_load(self):
        cpp_file = self.workspace_root / "runtime/native_queue_collapse_runtime.cpp"
        
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
                print("[Native Compile] Compiling native queue DLL...")
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
                    print(f"[Native Compile] Queue DLL compiled successfully: {self.lib_path}")
            except Exception as e:
                print(f"[Native Compile] Warning: Could not run g++: {e}")
                
        if self.lib_path.exists():
            try:
                self.dll = ctypes.CDLL(str(self.lib_path))
                
                self.dll.init_native_queue.argtypes = []
                self.dll.init_native_queue.restype = None
                
                self.dll.enqueue_native_request.argtypes = [ctypes.c_int, ctypes.c_int]
                self.dll.enqueue_native_request.restype = ctypes.c_int
                
                self.dll.dequeue_native_request.argtypes = []
                self.dll.dequeue_native_request.restype = ctypes.c_int
                
                self.dll.get_native_queue_depth.argtypes = []
                self.dll.get_native_queue_depth.restype = ctypes.c_int
                
                self.dll.arbitrate_next_slot.argtypes = []
                self.dll.arbitrate_next_slot.restype = ctypes.c_int
                
                self.dll.init_native_queue()
                self.compiled = True
                print("[Native Load] Queue DLL successfully loaded via ctypes.")
            except Exception as e:
                print(f"[Native Load] Warning: Load Queue DLL failed: {e}")
                self.compiled = False
        else:
            self.compiled = False

    def enqueue_request(self, request_id: int, priority: int) -> bool:
        if self.dll and self.compiled:
            return self.dll.enqueue_native_request(ctypes.c_int(request_id), ctypes.c_int(priority)) == 1
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in enqueue_request!")

    def dequeue_request(self) -> int:
        if self.dll and self.compiled:
            return int(self.dll.dequeue_native_request())
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in dequeue_request!")

    def get_queue_depth(self) -> int:
        if self.dll and self.compiled:
            return int(self.dll.get_native_queue_depth())
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in get_queue_depth!")

    def arbitrate_next_slot(self) -> int:
        if self.dll and self.compiled:
            return int(self.dll.arbitrate_next_slot())
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in arbitrate_next_slot!")
