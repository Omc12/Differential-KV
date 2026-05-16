"""
memory/bandwidth_pressure_monitor.py

Monitors GPU bandwidth pressure and triggers sparsification or offloading.
Ensures throughput doesn't collapse under high memory traffic.
"""

import torch
import time

class BandwidthPressureMonitor:
    def __init__(self, high_threshold: float = 0.85, low_threshold: float = 0.4):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.pressure_history = []

    def check_pressure(self, traffic_mb: float, peak_bandwidth_mbs: float):
        """
        Checks current bandwidth utilization against peak hardware capability.
        Returns 'HIGH', 'NORMAL', or 'LOW'.
        """
        utilization = traffic_mb / peak_bandwidth_mbs
        self.pressure_history.append(utilization)
        
        if utilization > self.high_threshold:
            return "HIGH"
        elif utilization < self.low_threshold:
            return "LOW"
        return "NORMAL"

    def get_dynamic_sparsification_factor(self):
        """
        Suggests a sparsification factor based on recent bandwidth pressure.
        If pressure is high, increase sparsity (lower density).
        """
        if not self.pressure_history: return 1.0
        
        avg_pressure = sum(self.pressure_history[-10:]) / len(self.pressure_history[-10:])
        if avg_pressure > self.high_threshold:
            return 0.5 # Request 2x more sparsity
        elif avg_pressure < self.low_threshold:
            return 1.5 # Request 1.5x more density (better quality)
        return 1.0

    def get_monitor_report(self):
        """Returns average and peak bandwidth utilization stats."""
        if not self.pressure_history: return {}
        return {
            "avg_utilization": sum(self.pressure_history) / len(self.pressure_history),
            "peak_utilization": max(self.pressure_history),
            "pressure_events": sum(1 for p in self.pressure_history if p > self.high_threshold)
        }
