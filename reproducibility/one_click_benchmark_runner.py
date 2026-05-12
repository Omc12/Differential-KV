"""
reproducibility/one_click_benchmark_runner.py

Main orchestrator for executing the full Differential KV validation suite.
Automatically runs benchmarks, profiles, and generates preliminary results.
"""

import subprocess
import os
import sys
import json
import time

class OneClickRunner:
    def __init__(self):
        self.benchmarks = [
            "benchmarks/open_longbench_eval.py",
            "benchmarks/open_gsm8k_eval.py",
            "benchmarks/open_humaneval_eval.py",
            "benchmarks/open_needle_eval.py"
        ]
        self.output_dir = "results/phase38/full_run"
        os.makedirs(self.output_dir, exist_ok=True)

    def run_benchmark(self, script_path: str):
        print(f"[{time.strftime('%H:%M:%S')}] Starting: {script_path}")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[{time.strftime('%H:%M:%S')}] Completed: {script_path}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error running {script_path}: {e}")
            print(e.stderr)
            return None

    def run_all(self):
        print("="*60)
        print("Differential KV: Phase 38 Open Frontier Validation Runner")
        print("="*60)
        
        summary = {}
        for bench in self.benchmarks:
            output = self.run_benchmark(bench)
            summary[bench] = "Success" if output else "Failed"
        
        print("\n" + "="*60)
        print("Validation Summary:")
        for bench, status in summary.items():
            print(f"  {bench}: {status}")
        print("="*60)

if __name__ == "__main__":
    runner = OneClickRunner()
    runner.run_all()
