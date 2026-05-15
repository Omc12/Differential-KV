
"""
resume_reconstruction_20_6A_validation.py

PHASE 20.6A: PPOSAH - Precision Path Optimization & Symbolic Alignment Hardening
RESUME SCRIPT - Resumes from 16k json_exact.
"""

import torch
import json
import time
import os
from transformers import AutoTokenizer
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.sps_precision_suite import SPSPrecisionSuite

# Resolvers
from runtime.pposah_resolver import PPOSAHResolver
from runtime.sps_resolver import SPSResolver
from runtime.alfsr_resolver import ALFSRResolver
from runtime.adaptive_salience_resolver import AdaptiveSalienceResolver

# Import the original execution functions to ensure consistency
from run_reconstruction_20_6A_validation import execute_run, RESULTS_DIR, REPORTS_DIR

class ResumeValidationRunner20_6A:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.suite = SPSPrecisionSuite(tokenizer)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # OPEN IN APPEND MODE
        self.files = {
            "raw_anchor_alignment.jsonl": open(os.path.join(RESULTS_DIR, "raw_anchor_alignment.jsonl"), "a", encoding="utf-8"),
            "raw_precision_locality.jsonl": open(os.path.join(RESULTS_DIR, "raw_precision_locality.jsonl"), "a", encoding="utf-8"),
            "raw_gpu_stalls.jsonl": open(os.path.join(RESULTS_DIR, "raw_gpu_stalls.jsonl"), "a", encoding="utf-8"),
            "raw_softmax_reuse.jsonl": open(os.path.join(RESULTS_DIR, "raw_softmax_reuse.jsonl"), "a", encoding="utf-8"),
            "raw_suffix_fidelity.jsonl": open(os.path.join(RESULTS_DIR, "raw_suffix_fidelity.jsonl"), "a", encoding="utf-8"),
            "raw_entropy_balance.jsonl": open(os.path.join(RESULTS_DIR, "raw_entropy_balance.jsonl"), "a", encoding="utf-8"),
            "raw_symbolic_drift.jsonl": open(os.path.join(RESULTS_DIR, "raw_symbolic_drift.jsonl"), "a", encoding="utf-8"),
            "raw_token_generation.jsonl": open(os.path.join(RESULTS_DIR, "raw_token_generation.jsonl"), "a", encoding="utf-8"),
        }
        self.wallclock_log = open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "a", encoding="utf-8")

    def _log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self.wallclock_log.write(line + "\n")
        self.wallclock_log.flush()

    def _write(self, fname, obj):
        self.files[fname].write(json.dumps(obj) + "\n")
        self.files[fname].flush()

    def run(self):
        modes = ["dense", "sparse_baseline", "alfsr_20_5", "sps_20_6", "pposah_20_6a"]
        contexts = [16384] # Only 16k remaining
        domains = ["hex_sequence", "api_key_complex", "suffix_integrity", "json_exact", "casing_adversarial"]
        
        # RESUME LOGIC
        # Already finished: hex_sequence, api_key_complex, suffix_integrity at 16k.
        # Interrupted at json_exact.
        
        domains_to_run = ["json_exact", "casing_adversarial"]
        
        self._log("RESUMING VALIDATION FROM 16K JSON_EXACT")

        for ctx in contexts:
            for domain in domains_to_run:
                test_case = self.suite.create_case(domain, ctx)
                for mode in modes:
                    self._log(f"Running mode={mode} ctx={ctx} domain={domain}")
                    resolver = self._make_resolver(mode, ctx)
                    try:
                        res = execute_run(self.model, self.tokenizer, test_case, mode, resolver)
                        
                        # Write telemetry
                        self._write("raw_gpu_stalls.jsonl", {"mode": mode, "ctx": ctx, "overhead": res["sync_overhead"]})
                        self._write("raw_softmax_reuse.jsonl", {"mode": mode, "ctx": ctx, "softmax_calls": res["softmax_calls"]})
                        self._write("raw_suffix_fidelity.jsonl", {"mode": mode, "ctx": ctx, "domain": domain, "fidelity": res["fidelity"]})
                        
                        if mode == "pposah_20_6a" and res["resolver_telemetry"]:
                            for t in res["resolver_telemetry"]:
                                self._write("raw_entropy_balance.jsonl", {"mode": mode, "ctx": ctx, "entropy": t["entropy_nats"]})
                                self._write("raw_symbolic_drift.jsonl", {"mode": mode, "ctx": ctx, "risk": t["drift_risk"]})
                        
                        self._log(f"  Result: EM={res['exact_match']} Fid={res['fidelity']:.3f} TPS={res['tps']:.1f}")
                    except Exception as e:
                        self._log(f"  [ERROR] {e}")
        
        self.close()

    def _make_resolver(self, mode, ctx):
        budget = ctx // 2
        if mode == "dense": return None
        if mode == "sparse_baseline": return AdaptiveSalienceResolver(self.tokenizer, anchor_budget=budget)
        if mode == "alfsr_20_5": return ALFSRResolver(self.tokenizer, anchor_budget=budget)
        if mode == "sps_20_6": return SPSResolver(self.tokenizer, anchor_budget=budget)
        if mode == "pposah_20_6a": return PPOSAHResolver(self.tokenizer, anchor_budget=budget)
        return None

    def close(self):
        for f in self.files.values(): f.close()
        self.wallclock_log.close()

if __name__ == "__main__":
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    runner = ResumeValidationRunner20_6A(model, tokenizer)
    runner.run()
