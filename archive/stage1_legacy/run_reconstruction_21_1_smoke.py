
import os
import json
import time
import subprocess
import torch
from typing import List

class SRLSmokeRunner:
    """
    PHASE 21.1: SRL (Symbolic Recall Legitimacy) FOCUSED SMOKE TEST.
    Target time: < 10-15 minutes.
    Total tests: 12
    """
    def __init__(self):
        # 12 focused legitimacy tests
        self.tests = [
            # 1. Correct Symbolic Recall (Legitimacy verification)
            {"mode": "srl_21_1", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            
            # 2. Irrelevant Recall Suppression (False Recall prevention)
            {"mode": "srl_21_1", "ctx": 8192, "domain": "hex_sequence", "len": 64},
            
            # 3. Multi-candidate Symbolic Routing
            {"mode": "srl_21_1", "ctx": 4096, "domain": "structured_id", "len": 64},
            
            # 4. Entropy Preservation (Verified via metrics)
            {"mode": "srl_21_1", "ctx": 4096, "domain": "json_exact", "len": 64},
            
            # 5. Delayed Symbolic Recall (Wait and see)
            {"mode": "srl_21_1", "ctx": 8192, "domain": "activation_code", "len": 64},
            
            # 6. Random Symbolic Confusion Stress
            {"mode": "srl_21_1", "ctx": 4096, "domain": "adversarial_delimiters", "len": 64},
            
            # Baselines for comparison
            {"mode": "hsha_21_0", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            {"mode": "hsha_21_0", "ctx": 8192, "domain": "hex_sequence", "len": 64},
            {"mode": "sabeaf_20_8", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            {"mode": "dense", "ctx": 4096, "domain": "api_key_complex", "len": 64},
            
            # Additional SRL tests
            {"mode": "srl_21_1", "ctx": 4096, "domain": "propagation_chain", "len": 128},
            {"mode": "srl_21_1", "ctx": 8192, "domain": "json_reconstruction", "len": 64},
        ]
        self.active_process = None

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def execute_single_run(self, mode: str, ctx: int, domain: str, prop_len: int):
        cmd = [
            "python", 
            "d:\\Codes\\Projects\\Differential KV\\scratch\\single_run_21_1.py",
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
        self._log("Starting HSHA Phase 21.1 SRL Focused Smoke Test...")
        self._log(f"Policy: LIGHTWEIGHT (Max {len(self.tests)} tests)")
        
        # Clear VRAM
        torch.cuda.empty_cache()
        
        start_total = time.time()
        
        for i, test in enumerate(self.tests):
            self._log(f"[{i+1}/{len(self.tests)}] {test['mode']} | {test['domain']} | {test['ctx']} ctx")
            self.execute_single_run(test['mode'], test['ctx'], test['domain'], test['len'])
            
        duration = (time.time() - start_total) / 60
        self._log(f"SRL Smoke Test Complete. Total duration: {duration:.2f} minutes.")

if __name__ == "__main__":
    runner = SRLSmokeRunner()
    runner.run()
