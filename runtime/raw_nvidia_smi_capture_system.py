import os
import subprocess
import time
import threading
from pathlib import Path

class RawNvidiaSmiCaptureSystem:
    """
    RHD Phase 41.4.6 — Raw NVIDIA-SMI Capture System.
    Captures raw GPU telemetry directly from nvidia-smi and nvidia-smi dmon.
    Saves raw logs with timestamp alignment. No interpretation, no synthesis.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.log_dir = self.workspace_root / "telemetry/stage3b/phase_41_4_6_rhd"
        self.smi_log_path = self.log_dir / "raw_nvidia_smi.log"
        self.dmon_log_path = self.log_dir / "raw_nvidia_smi_dmon.log"
        
        self.smi_proc = None
        self.dmon_proc = None
        self.running = False

    def start(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.running = True
        
        # Open log files in write mode (clearing old files)
        self.smi_file = open(self.smi_log_path, "w", encoding="utf-8")
        self.dmon_file = open(self.dmon_log_path, "w", encoding="utf-8")
        
        # Start nvidia-smi dmon subprocess
        # Windows command line, direct logging
        try:
            self.dmon_proc = subprocess.Popen(
                ["nvidia-smi", "dmon"],
                stdout=self.dmon_file,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
        except Exception as e:
            print(f"[NVIDIA-SMI Capture] Warning: Could not start nvidia-smi dmon: {e}")
            
        # Start standard nvidia-smi continuous query
        # Querying timestamp, utilization, memory, power draw, clocks, temperature, and PCIe gen/width
        try:
            query_fields = (
                "timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,"
                "power.draw,clocks.gr,clocks.mem,temperature.gpu,"
                "pcie.link.gen.current,pcie.link.width.current"
            )
            self.smi_proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv", "-l", "1"],
                stdout=self.smi_file,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
        except Exception as e:
            print(f"[NVIDIA-SMI Capture] Warning: Could not start nvidia-smi query: {e}")
            
        print("[NVIDIA-SMI Capture] Raw telemetry loggers started.")

    def stop(self):
        self.running = False
        
        # Stop dmon process
        if self.dmon_proc:
            try:
                self.dmon_proc.terminate()
                self.dmon_proc.wait(timeout=2)
            except Exception:
                try:
                    self.dmon_proc.kill()
                except Exception:
                    pass
            self.dmon_proc = None
            
        # Stop query process
        if self.smi_proc:
            try:
                self.smi_proc.terminate()
                self.smi_proc.wait(timeout=2)
            except Exception:
                try:
                    self.smi_proc.kill()
                except Exception:
                    pass
            self.smi_proc = None
            
        # Close file handles
        if hasattr(self, "smi_file") and self.smi_file:
            self.smi_file.close()
        if hasattr(self, "dmon_file") and self.dmon_file:
            self.dmon_file.close()
            
        print("[NVIDIA-SMI Capture] Raw telemetry loggers stopped.")
