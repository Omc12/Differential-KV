
import os
import json
import time
import subprocess
from typing import List

# Setup results directory
RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
os.makedirs(RESULTS_DIR, exist_ok=True)

class OrchestratedValidationRunner20_8_16k_Stress:
    """
    Phase 20.8 LIMITED 16k STRESS VALIDATION.
    Observes symbolic collapse boundaries at the edge of VRAM stability.
    Runs 10 critical stress cases only.
    """
    def __init__(self):
        # 16k Stress Matrix (Highly targeted)
        self.test_matrix = [
            {"mode": "sabeaf_20_8", "domain": "propagation_chain", "prop_len": 128},
            {"mode": "dense", "domain": "propagation_chain", "prop_len": 128},
            {"mode": "sabeaf_20_8", "domain": "delimiter_integrity", "prop_len": 128},
            {"mode": "sabeaf_20_8", "domain": "hex_sequence", "prop_len": 128},
            {"mode": "pposah_20_6a", "domain": "propagation_chain", "prop_len": 128},
            {"mode": "spslrif_20_7", "domain": "propagation_chain", "prop_len": 128},
            {"mode": "sabeaf_20_8", "domain": "api_key_complex", "prop_len": 64},
            {"mode": "sabeaf_20_8", "domain": "structured_id", "prop_len": 128},
            {"mode": "dense", "domain": "delimiter_integrity", "prop_len": 128},
            {"mode": "sabeaf_20_8", "domain": "propagation_chain", "prop_len": 64},
        ]
        self.active_process = None

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        with open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "a") as f:
            f.write(f"[{timestamp}] [STRESS-16k] {msg}\n")

    def _kill_stragglers(self):
        try:
            if self.active_process and self.active_process.poll() is None:
                self.active_process.terminate()
                self.active_process.wait(timeout=5)
        except:
            pass

    def execute_single_run(self, mode: str, ctx: int, domain: str, prop_len: int):
        cmd = [
            "python", 
            "d:\\Codes\\Projects\\Differential KV\\scratch\\single_run_20_8.py",
            "--mode", mode,
            "--ctx", str(ctx),
            "--domain", domain,
            "--prop_len", str(prop_len)
        ]
        try:
            self.active_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = self.active_process.communicate(timeout=450)
                if self.active_process.returncode != 0:
                    self._log(f"  ERROR in {mode}/{domain}: {stderr}")
                else:
                    summary = stdout.strip().split("\n")[-1]
                    self._log(f"  {summary}")
            except subprocess.TimeoutExpired:
                self.active_process.kill()
                self._log(f"  TIMEOUT in {mode}/{domain} (Killed)")
        except Exception as e:
            self._log(f"  CRITICAL FAILURE: {e}")
        finally:
            self.active_process = None

    def run(self):
        try:
            self._log("Starting Phase 20.8 LIMITED 16k STRESS VALIDATION (10 tests)...")
            total_runs = len(self.test_matrix)
            current_run = 0

            for test in self.test_matrix:
                current_run += 1
                self._log(f"[{current_run}/{total_runs}] Stress Launch: mode={test['mode']} ctx=16384 domain={test['domain']} prop_len={test['prop_len']}")
                self.execute_single_run(test['mode'], 16384, test['domain'], test['prop_len'])

            self._log("Generating 16k Stress Reports...")
            subprocess.run(["python", "d:\\Codes\\Projects\\Differential KV\\scratch\\generate_reports_20_8.py"])
            self._log("Phase 20.8 16k Stress Validation Complete.")
        except KeyboardInterrupt:
            self._log("Stress test interrupted. Cleaning up...")
        finally:
            self._kill_stragglers()

if __name__ == "__main__":
    runner = OrchestratedValidationRunner20_8_16k_Stress()
    runner.run()
