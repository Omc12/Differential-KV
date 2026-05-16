import subprocess
import time
import os
import json
import torch
import sys
from concurrent.futures import ThreadPoolExecutor

class PhysicalRealityAuditor:
    def __init__(self, duration=3600):
        self.duration = duration
        self.output_dir = "telemetry/stage2/phase_38_5_tft/"
        self.is_running = True
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.dmon_log = os.path.join(self.output_dir, "raw_nvidia_smi_dmon.log")
        # Clear previous fake logs
        if os.path.exists(self.dmon_log):
            os.remove(self.dmon_log)

    def start_hardware_recording(self):
        """
        PASSIVE CAPTURE: Only records what nvidia-smi actually outputs.
        """
        print(f"[*] PASSIVE CAPTURE: Recording real nvidia-smi dmon -> {self.dmon_log}")
        # Piling the real output of nvidia-smi directly into the file.
        # No simulation loop.
        cmd = f"nvidia-smi dmon -s pmu > {self.dmon_log}"
        return subprocess.Popen(cmd, shell=True)

    def execute_gpu_work(self):
        """
        PHYSICAL PRESSURE: Actually spins the SMs and fills VRAM.
        """
        print("[*] PHYSICAL PRESSURE: Allocating 12GB VRAM and running MatMul kernels...")
        # Material VRAM residency
        try:
            weights = torch.randn((12 * 1024 * 1024 * 1024 // 2), dtype=torch.float16, device="cuda")
        except Exception as e:
            print(f"[!] VRAM Allocation failed: {e}")
            return

        start_time = time.time()
        while self.is_running and (time.time() - start_time < self.duration):
            # Heavy SM load
            a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
            b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
            c = torch.matmul(a, b)
            # Short sleep to prevent CPU-side lockup while keeping GPU saturated
            time.sleep(0.001)

    def monitor(self):
        start_time = time.time()
        print("\n" + "!"*50)
        print("REAL HARDWARE AUDIT — NO SYNTHETIC DATA")
        print("!"*50)
        print("This script will only produce data as fast as time passes.")
        print("If you run for 1 minute, you will get 60 samples. No more.")
        
        try:
            while self.is_running and (time.time() - start_time < self.duration):
                elapsed = time.time() - start_time
                # Get current VRAM for dashboard (REAL)
                mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"\r[REAL-TIME] Uptime: {elapsed:.1f}s | VRAM: {mem:.2f}GB | SM: ACTIVE", end="")
                time.sleep(1)
        except KeyboardInterrupt:
            self.is_running = False

    def run(self):
        telemetry_proc = self.start_hardware_recording()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.execute_gpu_work)
            self.monitor()
            
        if telemetry_proc:
            # Cleanly kill the background nvidia-smi
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(telemetry_proc.pid)], shell=True)
        print("\n[*] Real Audit Stopped. No synthetic data was generated.")

if __name__ == "__main__":
    auditor = PhysicalRealityAuditor(duration=3600)
    auditor.run()
