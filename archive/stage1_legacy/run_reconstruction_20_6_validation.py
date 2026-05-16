"""
run_reconstruction_20_6_validation.py

PHASE 20.6: SPS - Symbolic Precision Stabilization
Validation harness.

Runs 6 modes across 4k, 8k, and 16k context matrix.
Exports mandatory telemetry for symbolic drift and precision.
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
from validation.noisy_context_injector import NoisyContextInjector

# Resolvers
from runtime.sps_resolver import SPSResolver
from runtime.alfsr_resolver import ALFSRResolver
from runtime.attention_steering_resolver import AttentionSteeringResolver
from runtime.adaptive_salience_resolver import AdaptiveSalienceResolver
from runtime.calibrated_memory_resolver import CalibratedMemoryResolver

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_6"
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
        
        # Guide decoder
        if resolver is not None and hasattr(resolver, "guide_decoder"):
            if isinstance(resolver, (ALFSRResolver, AttentionSteeringResolver, SPSResolver)):
                guided_logits = resolver.guide_decoder(raw_logits, attentions)
            else:
                guided_logits = resolver.guide_decoder(raw_logits)
        else:
            guided_logits = raw_logits
            
        probs = torch.softmax(guided_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_id = token.item()
        generated_tokens.append(token_id)
        
        if resolver is not None and hasattr(resolver, "record_generated_token"):
            resolver.record_generated_token(token_id, guided_logits)
            
        curr_input = token
        curr_pos += 1
        if token_id == tokenizer.eos_token_id:
            break
            
    duration = time.perf_counter() - start_gen
    tps = len(generated_tokens) / duration if duration > 0 else 0.0
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Exact Match
    exact_match = needle.lower() in output_text.lower()
    
    # Fidelity Metrics (Levenshtein based)
    import editdistance
    def fidelity(output, target):
        if not target: return 0.0
        target = target.lower()
        output = output.lower()
        
        # If target is in output, perfect fidelity
        if target in output:
            return 1.0
            
        # Otherwise, find the best matching substring of length ~len(target)
        best_dist = len(target)
        t_len = len(target)
        
        # Sliding window (optimization: only check near where we expect it)
        # For validation, we'll just check the whole thing if it's not too long
        for i in range(len(output) - t_len + 1):
            window = output[i:i+t_len]
            dist = editdistance.eval(window, target)
            if dist < best_dist:
                best_dist = dist
                
        return max(0.0, 1.0 - (best_dist / t_len))

    fid_score = fidelity(output_text, needle)
    
    # Extract SPS trace if available
    sps_trace = getattr(resolver, "precision_trace", [])
    
    return {
        "mode": mode,
        "ctx": input_ids.shape[1],
        "domain": test_case["domain"],
        "needle": needle,
        "output": output_text,
        "exact_match": exact_match,
        "fidelity": fid_score,
        "tps": tps,
        "sps_trace": sps_trace,
        "num_tokens": len(generated_tokens)
    }

class ValidationRunner20_6:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.suite = SPSPrecisionSuite(tokenizer)
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        self.files = {
            "raw_symbolic_precision.jsonl": open(os.path.join(RESULTS_DIR, "raw_symbolic_precision.jsonl"), "w", encoding="utf-8"),
            "raw_suffix_stability.jsonl": open(os.path.join(RESULTS_DIR, "raw_suffix_stability.jsonl"), "w", encoding="utf-8"),
            "raw_drift_predictions.jsonl": open(os.path.join(RESULTS_DIR, "raw_drift_predictions.jsonl"), "w", encoding="utf-8"),
            "raw_entropy_precision_balance.jsonl": open(os.path.join(RESULTS_DIR, "raw_entropy_precision_balance.jsonl"), "w", encoding="utf-8"),
            "raw_decoder_diversity.jsonl": open(os.path.join(RESULTS_DIR, "raw_decoder_diversity.jsonl"), "w", encoding="utf-8"),
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
        modes = ["dense", "sparse_baseline", "dtascc_19_7", "aascsi_20_3", "alfsr_20_5", "sps_20_6"]
        contexts = [4096, 8192, 16384]
        domains = ["hex_sequence", "api_key_complex", "casing_adversarial", "json_exact", "suffix_integrity"]
        
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
                        self._write("raw_symbolic_precision.jsonl", {
                            "mode": mode, "ctx": ctx, "domain": domain,
                            "exact_match": res["exact_match"], "fidelity": res["fidelity"], "tps": res["tps"]
                        })
                        
                        if mode == "sps_20_6" and res["sps_trace"]:
                            for t in res["sps_trace"]:
                                t.update({"mode": mode, "ctx": ctx, "domain": domain})
                                self._write("raw_drift_predictions.jsonl", t)
                                self._write("raw_entropy_precision_balance.jsonl", {
                                    "step": t["step"], "entropy": t["entropy_nats"], "stab": t["stab_factor"]
                                })
                                
                        self._log(f"  Result: EM={res['exact_match']} Fid={res['fidelity']:.3f} TPS={res['tps']:.1f}")
                    except Exception as e:
                        self._log(f"  [ERROR] {e}")
                        
        self._generate_reports(all_results)

    def _make_resolver(self, mode, ctx):
        budget = ctx // 2
        if mode == "dense": return None
        if mode == "sparse_baseline": return AdaptiveSalienceResolver(self.tokenizer, anchor_budget=budget)
        if mode == "dtascc_19_7": return CalibratedMemoryResolver(anchor_budget=budget)
        if mode == "aascsi_20_3": 
            r = AttentionSteeringResolver(self.tokenizer, anchor_budget=budget)
            r.logit_bias_strength = 15.0
            return r
        if mode == "alfsr_20_5": return ALFSRResolver(self.tokenizer, anchor_budget=budget)
        if mode == "sps_20_6": return SPSResolver(self.tokenizer, anchor_budget=budget)
        return None

    def _generate_reports(self, results):
        self._log("Generating reports...")
        
        # Report 1: Symbolic Precision
        lines = ["# Phase 20.6: Symbolic Precision Performance\n\n"]
        lines.append("| Mode | Avg Fid (4k) | Avg Fid (8k) | Avg Fid (16k) | Total EM |\n")
        lines.append("|---|---|---|---|---|\n")
        
        for mode in ["dense", "sparse_baseline", "dtascc_19_7", "aascsi_20_3", "alfsr_20_5", "sps_20_6"]:
            fids = {4096: [], 8192: [], 16384: []}
            ems = 0
            for r in results:
                if r["mode"] == mode:
                    fids[r["ctx"]].append(r["fidelity"])
                    if r["exact_match"]: ems += 1
            
            line = f"| {mode} | "
            for c in [4096, 8192, 16384]:
                avg = sum(fids[c])/len(fids[c]) if fids[c] else 0.0
                line += f"{avg:.3f} | "
            line += f"{ems} |\n"
            lines.append(line)
            
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_symbolic_precision.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        # Report 2: Drift Risk & Entropy
        lines = ["# Phase 20.6: Drift Risk & Entropy Balance\n\n"]
        lines.append("| Mode | Avg Entropy (Nats) | Drift Rejections | Collapse Count |\n")
        lines.append("|---|---|---|---|---|\n")
        
        # We only really have this for SPS
        sps_entries = [r for r in results if r["mode"] == "sps_20_6"]
        if sps_entries:
            all_traces = []
            for r in sps_entries: all_traces.extend(r["sps_trace"])
            
            if all_traces:
                avg_entropy = sum(t["entropy_nats"] for t in all_traces) / len(all_traces)
                collapses = sum(1 for t in all_traces if t["is_collapsed"])
                line = f"| sps_20_6 | {avg_entropy:.4f} | N/A | {collapses} |\n"
                lines.append(line)
                
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_6_drift_entropy.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        self._log("All reports generated.")

    def close(self):
        for f in self.files.values():
            f.close()
        self.wallclock_log.close()

if __name__ == "__main__":
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner20_6(model, tokenizer)
    try:
        runner.run()
    finally:
        runner.close()
