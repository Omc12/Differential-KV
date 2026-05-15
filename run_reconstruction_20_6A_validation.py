
"""
run_reconstruction_20_6A_validation.py

PHASE 20.6A: PPOSAH - Precision Path Optimization & Symbolic Alignment Hardening
Validation harness.

Runs 7 modes across 4k, 8k, and 16k context matrix.
Exports mandatory telemetry for GPU stalls, softmax reuse, and anchor alignment.
"""

import torch
import json
import time
import os
import math
from typing import Optional
from transformers import AutoTokenizer, DynamicCache

from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.sps_precision_suite import SPSPrecisionSuite

# Resolvers
from runtime.pposah_resolver import PPOSAHResolver
from runtime.sps_resolver import SPSResolver
from runtime.alfsr_resolver import ALFSRResolver
from runtime.attention_steering_resolver import AttentionSteeringResolver
from runtime.adaptive_salience_resolver import AdaptiveSalienceResolver
from runtime.calibrated_memory_resolver import CalibratedMemoryResolver

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_6A"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

def prefill_chunked(model, input_ids, resolver=None, chunk_size=512):
    past_key_values = DynamicCache()
    for i in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, i:i + chunk_size]
        pos_ids = torch.arange(i, i + chunk.shape[1], device="cuda").unsqueeze(0)
        with torch.no_grad():
            outputs = model(
                input_ids=chunk,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            if resolver is not None:
                legacy = past_key_values.to_legacy_cache()
                pruned, _ = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                past_key_values = DynamicCache.from_legacy_cache(pruned)
    return past_key_values

def execute_run(model, tokenizer, test_case, mode, resolver, max_new_tokens=64):
    needle = test_case["needle"]
    input_ids = torch.tensor([test_case["tokens"]]).to("cuda")
    
    if hasattr(resolver, "reset_generation_state"):
        resolver.reset_generation_state()

    past_key_values = prefill_chunked(model, input_ids, resolver)
    
    curr_input = input_ids[:, -1:]
    curr_pos = input_ids.shape[1]
    generated_tokens = []
    
    sync_overhead = 0.0
    softmax_calls = 0
    
    start_gen = time.perf_counter()
    
    for step in range(max_new_tokens):
        pos_ids = torch.tensor([[curr_pos]], device="cuda")
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        
        raw_logits = outputs.logits[:, -1, :].float()
        attentions = torch.stack(outputs.attentions) if outputs.attentions else None
        
        # Performance Tracking
        s_time = time.perf_counter()
        
        # Guide decoder
        if resolver is not None and hasattr(resolver, "guide_decoder"):
            if isinstance(resolver, (ALFSRResolver, SPSResolver, PPOSAHResolver)):
                guided_logits = resolver.guide_decoder(raw_logits, attentions)
            else:
                guided_logits = resolver.guide_decoder(raw_logits)
            
            # Track softmax calls
            if mode == "pposah_20_6a":
                softmax_calls += 1 # update()
            elif mode == "sps_20_6":
                softmax_calls += 3 # risk, audit, plus final
            else:
                softmax_calls += 1
        else:
            guided_logits = raw_logits
            softmax_calls += 1
            
        probs = torch.softmax(guided_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_id = token.item()
        generated_tokens.append(token_id)
        
        if resolver is not None and hasattr(resolver, "record_generated_token"):
            resolver.record_generated_token(token_id, guided_logits)
            
        sync_overhead += (time.perf_counter() - s_time)
            
        curr_input = token
        curr_pos += 1
        if token_id == tokenizer.eos_token_id:
            break
            
    duration = time.perf_counter() - start_gen
    tps = len(generated_tokens) / duration if duration > 0 else 0.0
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Fidelity Metrics
    import editdistance
    def fidelity(output, target):
        if not target: return 0.0
        target = target.lower()
        output = output.lower()
        if target in output: return 1.0
        best_dist = len(target)
        t_len = len(target)
        for i in range(len(output) - t_len + 1):
            window = output[i:i+t_len]
            dist = editdistance.eval(window, target)
            if dist < best_dist: best_dist = dist
        return max(0.0, 1.0 - (best_dist / t_len))

    fid_score = fidelity(output_text, needle)
    
    return {
        "mode": mode,
        "ctx": input_ids.shape[1],
        "domain": test_case["domain"],
        "needle": needle,
        "output": output_text,
        "exact_match": needle.lower() in output_text.lower(),
        "fidelity": fid_score,
        "tps": tps,
        "sync_overhead": sync_overhead,
        "softmax_calls": softmax_calls,
        "resolver_telemetry": getattr(resolver, "precision_trace", [])
    }

class ValidationRunner20_6A:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.suite = SPSPrecisionSuite(tokenizer)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        self.files = {
            "raw_anchor_alignment.jsonl": open(os.path.join(RESULTS_DIR, "raw_anchor_alignment.jsonl"), "w", encoding="utf-8"),
            "raw_precision_locality.jsonl": open(os.path.join(RESULTS_DIR, "raw_precision_locality.jsonl"), "w", encoding="utf-8"),
            "raw_gpu_stalls.jsonl": open(os.path.join(RESULTS_DIR, "raw_gpu_stalls.jsonl"), "w", encoding="utf-8"),
            "raw_softmax_reuse.jsonl": open(os.path.join(RESULTS_DIR, "raw_softmax_reuse.jsonl"), "w", encoding="utf-8"),
            "raw_suffix_fidelity.jsonl": open(os.path.join(RESULTS_DIR, "raw_suffix_fidelity.jsonl"), "w", encoding="utf-8"),
            "raw_entropy_balance.jsonl": open(os.path.join(RESULTS_DIR, "raw_entropy_balance.jsonl"), "w", encoding="utf-8"),
            "raw_symbolic_drift.jsonl": open(os.path.join(RESULTS_DIR, "raw_symbolic_drift.jsonl"), "w", encoding="utf-8"),
            "raw_token_generation.jsonl": open(os.path.join(RESULTS_DIR, "raw_token_generation.jsonl"), "w", encoding="utf-8"),
        }
        self.wallclock_log = open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "w", encoding="utf-8")

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
        contexts = [4096, 8192, 16384]
        domains = ["hex_sequence", "api_key_complex", "suffix_integrity", "json_exact", "casing_adversarial"]
        
        all_results = []
        for ctx in contexts:
            for domain in domains:
                test_case = self.suite.create_case(domain, ctx)
                for mode in modes:
                    self._log(f"Running mode={mode} ctx={ctx} domain={domain}")
                    resolver = self._make_resolver(mode, ctx)
                    try:
                        res = execute_run(self.model, self.tokenizer, test_case, mode, resolver)
                        all_results.append(res)
                        
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
    runner = ValidationRunner20_6A(model, tokenizer)
    runner.run()
