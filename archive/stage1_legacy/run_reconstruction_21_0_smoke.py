
import os
import json
import time
import subprocess
import torch
from typing import List

class HSHASmokeRunner:
    """
    PHASE 21.x: FOCUSED ARCHITECTURAL SMOKE TEST.
    Aligns with the 'Validate Only When Necessary' policy.
    Target time: < 10-15 minutes.
    """
    def __init__(self):
        # Focused test matrix (10 tests)
        self.tests = [
            # 1. Exact Symbolic Recall (4k)
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "api_key_complex", "len": 128},
            
            # 2. Delimiter Topology Restoration (4k)
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "delimiter_integrity", "len": 64},
            
            # 3. Random Symbolic Persistence (4k)
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "random_symbolic_object", "len": 64},
            
            # 4. Multi-hop Symbolic Recall (4k)
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "multi_hop_recall", "len": 64},
            
            # 5. False Recall Suppression / Drift Handling (4k)
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "propagation_chain", "len": 64},
            
            # 6. Baseline Comparison (Minimal comparison)
            {"mode": "sabeaf_20_8", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            {"mode": "dense", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            
            # 7. 8k Stability Probe (Verify HSHA overhead at 8k)
            {"mode": "hsha_21_0", "ctx": 8192, "domain": "api_key_complex", "len": 64},
            {"mode": "dense", "ctx": 8192, "domain": "api_key_complex", "len": 64},
        ]
        self.active_process = None

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

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
                    self._log(f"  ERROR: {stderr}")
                else:
                    # Capture the result line
                    summary = [line for line in stdout.strip().split("\n") if "Result:" in line]
                    if summary:
                        self._log(f"  {summary[-1]}")
                    else:
                        self._log("  Execution finished (no summary found)")
            except subprocess.TimeoutExpired:
                self.active_process.kill()
                self._log(f"  TIMEOUT")
        except Exception as e:
            self._log(f"  FAILURE: {e}")
        finally:
            self.active_process = None

    def run(self):
        self._log("Starting HSHA Phase 21.0 Focused Smoke Test...")
        self._log(f"Policy: FOCUSED (Max {len(self.tests)} tests)")
        
        # Clear VRAM
        torch.cuda.empty_cache()
        
        start_total = time.time()
        
        for i, test in enumerate(self.tests):
            self._log(f"[{i+1}/{len(self.tests)}] {test['mode']} | {test['domain']} | {test['ctx']} ctx")
            self.execute_single_run(test['mode'], test['ctx'], test['domain'], test['len'])
            
        duration = (time.time() - start_total) / 60
        self._log(f"Smoke Test Complete. Total duration: {duration:.2f} minutes.")

if __name__ == "__main__":
    runner = HSHASmokeRunner()
    runner.run()
