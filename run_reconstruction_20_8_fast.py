
import os
import json
import time
import subprocess
from typing import List

# Setup results directory
RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
os.makedirs(RESULTS_DIR, exist_ok=True)

class OrchestratedValidationRunner20_8_Fast:
    """
    Phase 20.8 FAST RESEARCH VALIDATION MODE.
    Target duration: 30-45 minutes.
    Focuses on 120 critical test cases to allow rapid architectural iteration.
    """
    def __init__(self):
        # 8k is Primary Research Environment (Policy-Aligned)
        self.contexts = [8192]
        # 64 and 128 tokens for research iteration
        self.prop_lengths = [64, 128]
        # Core research domains
        self.domains = [
            "hex_sequence", 
            "api_key_complex", 
            "propagation_chain", 
            "delimiter_integrity", 
            "structured_id"
        ]
        # Current active modes (SABEAF vs baselines)
        self.modes = ["sabeaf_20_8", "spslrif_20_7", "pposah_20_6a", "dense"]
        self.active_process = None

    def _cleanup_telemetry(self):
        """Clears old telemetry to prevent result contamination."""
        self._log("Cleaning up old telemetry files...")
        for file in os.listdir(RESULTS_DIR):
            if file.endswith(".jsonl") or file.endswith(".log"):
                try:
                    os.remove(os.path.join(RESULTS_DIR, file))
                except Exception as e:
                    self._log(f"  Warning: Could not remove {file}: {e}")

    def _kill_stragglers(self):
        """Ensures no orphaned single_run processes are holding the GPU."""
        try:
            # On Windows, we look for python processes that might be hanging
            # We use taskkill to be thorough, filtering by command line is hard without psutil
            # But we can at least try to kill the active process if it exists
            if self.active_process and self.active_process.poll() is None:
                self.active_process.terminate()
                self.active_process.wait(timeout=5)
        except:
            pass

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        with open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

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
            # Using Popen for better control over termination
            self.active_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = self.active_process.communicate(timeout=480) # Increased for 128-token Tier
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

    def generate_all_reports(self):
        self._log("Generating Phase 20.8 Fast Validation Reports...")
        cmd = ["python", "d:\\Codes\\Projects\\Differential KV\\scratch\\generate_reports_20_8.py"]
        subprocess.run(cmd)

    def _get_completed_keys(self):
        """Returns a set of (mode, ctx, domain, prop_len) tuples already in results."""
        keys = set()
        path = os.path.join(RESULTS_DIR, "raw_symbolic_propagation.jsonl")
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        keys.add((d["mode"], d["ctx"], d["domain"], d["prop_len"]))
                    except: continue
        return keys

    def run(self):
        try:
            # Policy Change: Disable cleanup for resume capability
            # self._cleanup_telemetry()
            self._log("Resuming Phase 20.8 FAST RESEARCH VALIDATION (8k Primary)...")
            completed = self._get_completed_keys()
            
            total_runs = len(self.contexts) * len(self.prop_lengths) * len(self.domains) * len(self.modes)
            current_run = 0

            for ctx in self.contexts:
                for prop_len in self.prop_lengths:
                    for domain in self.domains:
                        for mode in self.modes:
                            current_run += 1
                            if (mode, ctx, domain, prop_len) in completed:
                                continue
                                
                            self._log(f"[{current_run}/{total_runs}] Research Launch: mode={mode} ctx={ctx} domain={domain} prop_len={prop_len}")
                            self.execute_single_run(mode, ctx, domain, prop_len)

            self.generate_all_reports()
            self._log("Phase 20.8 FAST Validation Complete.")
        except KeyboardInterrupt:
            self._log("Validation interrupted by user. Cleaning up...")
        finally:
            self._kill_stragglers()

if __name__ == "__main__":
    runner = OrchestratedValidationRunner20_8_Fast()
    runner.run()
