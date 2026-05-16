
import os
import json
import time
import subprocess
from typing import List

# Setup results directory
RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
os.makedirs(RESULTS_DIR, exist_ok=True)

class OrchestratedValidationRunner20_8_Full:
    """
    Phase 20.8 FULL PUBLICATION VALIDATION SUITE.
    Target duration: 6+ hours.
    Exhaustive combinatorial benchmarking across all domains and modes.
    """
    def __init__(self):
        self.modes = ["dense", "sparse_baseline", "pposah_20_6a", "spslrif_20_7", "sabeaf_20_8"]
        self.contexts = [4096, 8192, 16384]
        # Full domain list including legacy tests
        self.domains = [
            "hex_sequence", "api_key_complex", "structured_id", 
            "propagation_chain", "delimiter_integrity", "json_exact",
            "activation_code", "adversarial_delimiters", "anchor_fragmentation",
            "json_reconstruction"
        ]
        self.prop_lengths = [64, 128]
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

    def generate_all_reports(self):
        self._log("Generating Phase 20.8 Full Scientific Reports...")
        cmd = ["python", "d:\\Codes\\Projects\\Differential KV\\scratch\\generate_reports_20_8.py"]
        subprocess.run(cmd)

    def run(self):
        try:
            self._cleanup_telemetry()
            self._log("Starting Phase 20.8 FULL PUBLICATION VALIDATION (Exhaustive)...")
            total_runs = len(self.contexts) * len(self.prop_lengths) * len(self.domains) * len(self.modes)
            current_run = 0

            for ctx in self.contexts:
                for prop_len in self.prop_lengths:
                    for domain in self.domains:
                        for mode in self.modes:
                            current_run += 1
                            self._log(f"[{current_run}/{total_runs}] Launching: mode={mode} ctx={ctx} domain={domain} prop_len={prop_len}")
                            self.execute_single_run(mode, ctx, domain, prop_len)

            self.generate_all_reports()
            self._log("Phase 20.8 FULL Validation Complete.")
        except KeyboardInterrupt:
            self._log("Validation interrupted by user. Cleaning up...")
        finally:
            self._kill_stragglers()

if __name__ == "__main__":
    runner = OrchestratedValidationRunner20_8_Full()
    runner.run()
