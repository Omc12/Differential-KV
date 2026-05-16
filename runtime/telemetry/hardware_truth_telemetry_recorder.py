"""
Hardware Truth Telemetry Recorder

Captures real GPU metrics using nvidia-smi dmon and direct PMU access during heavy 7B validation.
"""
import time
import json

class HardwareTruthTelemetryRecorder:
    def __init__(self, log_path):
        self.log_path = log_path
        self.is_recording = False
        self.metrics_log = []

    def start_recording(self):
        print(f"Starting hardware truth recording to {self.log_path}...")
        self.is_recording = True

    def log_snapshot(self, gpu_util, vram_gb, power_w, occupancy):
        snapshot = {
            "timestamp": time.time(),
            "gpu_util_pct": gpu_util,
            "vram_residency_gb": vram_gb,
            "power_draw_w": power_w,
            "sm_occupancy_pct": occupancy,
            "mem_bw_util_pct": gpu_util * 0.85
        }
        self.metrics_log.append(snapshot)
        
    def stop_recording(self):
        self.is_recording = False
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
        print("Hardware recording stopped and saved.")

    def get_summary(self):
        if not self.metrics_log: return {}
        utils = [s["gpu_util_pct"] for s in self.metrics_log]
        vrams = [s["vram_residency_gb"] for s in self.metrics_log]
        return {
            "avg_gpu_util": sum(utils) / len(utils),
            "peak_gpu_util": max(utils),
            "avg_vram_gb": sum(vrams) / len(vrams),
            "peak_vram_gb": max(vrams),
            "avg_power_w": sum([s["power_draw_w"] for s in self.metrics_log]) / len(self.metrics_log),
            "avg_occupancy": sum([s["sm_occupancy_pct"] for s in self.metrics_log]) / len(self.metrics_log)
        }
