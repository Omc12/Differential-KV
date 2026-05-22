"""
profiling/intervention_cost_analysis.py

Analyzes the computational cost of different stabilization interventions
(SAM update, ACTR repair, GRP lock, pulse scheduling).
"""

import time
import torch
from typing import Dict, Any, Callable

class InterventionCostAnalyzer:
    def __init__(self):
        self.costs = {}
        
    def measure_cost(self, name: str, func: Callable, *args, **kwargs):
        """
        Measures the execution time and CUDA overhead of a specific intervention.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.time()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end = time.time()
        cost_ms = (end - start) * 1000
        
        if name not in self.costs: self.costs[name] = []
        self.costs[name].append(cost_ms)
        
        return result

    def get_report(self) -> Dict[str, float]:
        report = {}
        for name, times in self.costs.items():
            report[name] = {
                "avg_ms": sum(times) / len(times),
                "max_ms": max(times),
                "total_calls": len(times)
            }
        return report
