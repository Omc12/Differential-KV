"""
Raw Hardware Trace Parser

Derives metrics ONLY from real telemetry traces (nvidia-smi dmon logs).
"""
import os
import json

class RawHardwareTraceParser:
    def __init__(self, dmon_log_path, polling_trace_path):
        self.dmon_log_path = dmon_log_path
        self.polling_trace_path = polling_trace_path

    def parse_dmon_log(self):
        """
        Parses raw nvidia-smi dmon output.
        Expects format: # gpu pwr gtemp mtemp sm mem enc dec mclk pclk
        """
        metrics = []
        if not os.path.exists(self.dmon_log_path):
            return metrics
            
        with open(self.dmon_log_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        metrics.append({
                            "pwr": float(parts[1]),
                            "sm": float(parts[5]),
                            "mem": float(parts[6]),
                            "mclk": float(parts[9])
                        })
                    except ValueError:
                        continue
        return metrics

    def parse_polling_trace(self):
        """
        Parses timestamped jsonl polling trace.
        """
        metrics = []
        if not os.path.exists(self.polling_trace_path):
            return metrics
            
        with open(self.polling_trace_path, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except:
                    continue
        return metrics

    def derive_truth_summary(self):
        dmon_data = self.parse_dmon_log()
        poll_data = self.parse_polling_trace()
        
        if not dmon_data or not poll_data:
            return None
            
        sm_utils = [d["sm"] for d in dmon_data]
        vrams = [p["vram_residency_gb"] for p in poll_data]
        durations = (poll_data[-1]["timestamp"] - poll_data[0]["timestamp"]) / 60.0
        
        return {
            "avg_sm_util": sum(sm_utils) / len(sm_utils),
            "peak_sm_util": max(sm_utils),
            "avg_vram_gb": sum(vrams) / len(vrams),
            "peak_vram_gb": max(vrams),
            "duration_minutes": durations,
            "sample_count": len(dmon_data),
            "utilization_stdev": self._stdev(sm_utils)
        }

    def _stdev(self, data):
        if len(data) < 2: return 0
        avg = sum(data) / len(data)
        var = sum((x - avg) ** 2 for x in data) / len(data)
        return var ** 0.5
