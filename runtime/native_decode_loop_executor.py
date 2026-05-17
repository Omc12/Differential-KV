import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path

class NativeDecodeLoopExecutor:
    """
    NDX Phase 42.1.5 — Native Decode Loop Executor.
    Compiles and links the native C++ code to execute hotpath decode loops.
    No silent Python fallbacks allowed.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        
        # Clean up any previously compiled dynamic DLLs
        for p in self.workspace_root.glob("runtime/native_decode_loop_executor_*.dll"):
            try:
                os.remove(p)
            except Exception:
                pass
                
        self.lib_path = self.workspace_root / f"runtime/native_decode_loop_executor_{int(time.time())}.dll"
        self.dll = None
        self.compiled = False
        
        self._compile_and_load()

    def _compile_and_load(self):
        cpp_file = self.workspace_root / "runtime/native_decode_loop_executor.cpp"
        
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
                print("[Native Compile] Compiling native C++ executor DLL...")
                # We can safely delete the DLL if it exists and is unlocked
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
                    print(f"[Native Compile] C++ DLL compiled successfully: {self.lib_path}")
                else:
                    print(f"[Native Compile] Warning: g++ failed: {res.stderr}")
            except Exception as e:
                print(f"[Native Compile] Warning: Could not run g++ compilation: {e}")
                
        # Load C++ DLL if present
        if self.lib_path.exists():
            try:
                self.dll = ctypes.CDLL(str(self.lib_path))
                
                self.dll.init_native_residency_table.argtypes = []
                self.dll.init_native_residency_table.restype = None
                
                self.dll.allocate_native_slot.argtypes = [ctypes.c_int, ctypes.c_int]
                self.dll.allocate_native_slot.restype = ctypes.c_int
                
                self.dll.release_native_slot.argtypes = [ctypes.c_int]
                self.dll.release_native_slot.restype = None
                
                self.dll.execute_native_decode_step.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int)
                ]
                self.dll.execute_native_decode_step.restype = None
                
                self.dll.get_active_native_slot_count.argtypes = []
                self.dll.get_active_native_slot_count.restype = ctypes.c_int
                
                self.dll.init_native_residency_table()
                self.compiled = True
                print("[Native Load] C++ DLL successfully loaded via ctypes.")
            except Exception as e:
                print(f"[Native Load] Warning: Load C++ DLL failed: {e}")
                self.compiled = False
        else:
            self.compiled = False

    def execute_decode_step(self, step: int, active_slots: int) -> tuple:
        """
        Executes a single decode step natively. Throws error on fallback.
        """
        if self.dll and self.compiled:
            latency = ctypes.c_float(0.0)
            launches = ctypes.c_int(0)
            self.dll.execute_native_decode_step(
                ctypes.c_int(step),
                ctypes.c_int(active_slots),
                ctypes.byref(latency),
                ctypes.byref(launches)
            )
            return latency.value, launches.value
            
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in execute_decode_step!")
