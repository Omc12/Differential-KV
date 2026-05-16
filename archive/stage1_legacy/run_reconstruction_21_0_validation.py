
import os
import json
import time
import subprocess
from typing import List

# Setup results directory
RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_21_0"
os.makedirs(RESULTS_DIR, exist_ok=True)

class OrchestratedValidationRunner21_0:
    """
    PHASE 21.0: HSHA (Hybrid Symbolic Hub Architecture) VALIDATION.
    Targets exact symbolic survival and structural preservation.
    """
    def __init__(self):
        self.contexts = [4096, 8192]
        self.prop_lengths = [64, 128]
        self.domains = [
            "api_key_complex",
            "delimiter_integrity",
            "propagation_chain",
            "random_symbolic_object",
            "multi_hop_recall" # Added for 21.0
        ]
        self.modes = ["hsha_21_0", "sabeaf_20_8", "spslrif_20_7", "dense"]
        self.active_process = None

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        with open(os.path.join(RESULTS_DIR, "wallclock_trace.log"), "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def execute_single_run(self, mode: str, ctx: int, domain: str, prop_len: int):
        cmd = [
            "python", 
            "d:\\Codes\\Projects\\Differential KV\\scratch\\single_run_21_0.py",
            "--mode", mode,
            "--ctx", str(ctx),
            "--domain", domain,
            "--prop_len", str(prop_len)
        ]
        try:
            self.active_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = self.active_process.communicate(timeout=600)
                if self.active_process.returncode != 0:
                    self._log(f"  ERROR in {mode}/{domain}: {stderr}")
                else:
                    summary = stdout.strip().split("\n")[-1]
                    self._log(f"  {summary}")
            except subprocess.TimeoutExpired:
                self.active_process.kill()
                self._log(f"  TIMEOUT in {mode}/{domain}")
        except Exception as e:
            self._log(f"  CRITICAL FAILURE: {e}")
        finally:
            self.active_process = None

    def run(self):
        self._log("Starting Phase 21.0 HSHA Architectural Validation...")
        total_runs = len(self.contexts) * len(self.prop_lengths) * len(self.domains) * len(self.modes)
        current_run = 0

        for ctx in self.contexts:
            for prop_len in self.prop_lengths:
                for domain in self.domains:
                    for mode in self.modes:
                        current_run += 1
                        self._log(f"[{current_run}/{total_runs}] Running: mode={mode} ctx={ctx} domain={domain} len={prop_len}")
                        self.execute_single_run(mode, ctx, domain, prop_len)

        self._log("Phase 21.0 Validation Complete.")

if __name__ == "__main__":
    runner = OrchestratedValidationRunner21_0()
    runner.run()
