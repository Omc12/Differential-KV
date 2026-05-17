import os
import sys
import time
import ctypes
import subprocess
from pathlib import Path

class NativeBatchResidencyScheduler:
    """
    NDX Phase 42.1.5 — Native Batch Residency Scheduler.
    Schedules active sessions completely natively.
    No silent Python scheduling fallbacks allowed.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        
        # Clean up any previously compiled dynamic DLLs
        for p in self.workspace_root.glob("runtime/native_batch_residency_scheduler_*.dll"):
            try:
                os.remove(p)
            except Exception:
                pass
                
        self.lib_path = self.workspace_root / f"runtime/native_batch_residency_scheduler_{int(time.time())}.dll"
        self.dll = None
        self.compiled = False
        
        self._compile_and_load()

    def _compile_and_load(self):
        cpp_file = self.workspace_root / "runtime/native_batch_residency_scheduler.cpp"
        
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
                print("[Native Compile] Compiling native scheduler DLL...")
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
                    print(f"[Native Compile] Scheduler DLL compiled successfully: {self.lib_path}")
            except Exception as e:
                print(f"[Native Compile] Warning: Could not run g++: {e}")
                
        if self.lib_path.exists():
            try:
                self.dll = ctypes.CDLL(str(self.lib_path))
                
                self.dll.init_native_scheduler.argtypes = []
                self.dll.init_native_scheduler.restype = None
                
                self.dll.schedule_native_session.argtypes = [ctypes.c_int]
                self.dll.schedule_native_session.restype = ctypes.c_int
                
                self.dll.evict_native_session.argtypes = [ctypes.c_int]
                self.dll.evict_native_session.restype = None
                
                self.dll.get_native_occupancy_rate.argtypes = []
                self.dll.get_native_occupancy_rate.restype = ctypes.c_float
                
                self.dll.init_native_scheduler()
                self.compiled = True
                print("[Native Load] Scheduler DLL successfully loaded via ctypes.")
            except Exception as e:
                print(f"[Native Load] Warning: Load Scheduler DLL failed: {e}")
                self.compiled = False
        else:
            self.compiled = False

    def schedule_session(self, session_id: str) -> int:
        if self.dll and self.compiled:
            session_hash = hash(session_id) & 0x7fffffff
            slot_id = self.dll.schedule_native_session(ctypes.c_int(session_hash))
            if slot_id < 0:
                raise RuntimeError(f"[NDX Overload] Failed to schedule session {session_id} - no active native slots.")
            return slot_id
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in schedule_session!")

    def evict_session(self, slot_id: int):
        if self.dll and self.compiled:
            self.dll.evict_native_session(ctypes.c_int(slot_id))
            return
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in evict_session!")

    def get_occupancy_rate(self) -> float:
        if self.dll and self.compiled:
            return float(self.dll.get_native_occupancy_rate())
        raise RuntimeError("[NDX Violation] Silent Python fallback occurred in get_occupancy_rate!")
