"""
run_reconstruction_20_5_validation.py

PHASE 20.5: ALFSR - Adaptive Low-Force Symbolic Recovery
Validation harness.

Runs ALL required modes and steering sweep across the full test matrix,
exports mandatory telemetry artifacts, and generates scientific reports.
"""

import torch
import json
import time
import os
import math
from typing import Optional

from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.multidomain_symbolic_suite import MultidomainSymbolicSuite
from validation.noisy_context_injector import NoisyContextInjector
from transformers import AutoTokenizer, DynamicCache

# Import all resolvers needed for mode comparison
from runtime.alfsr_resolver import ALFSRResolver
from runtime.attention_steering_resolver import AttentionSteeringResolver
from runtime.adaptive_salience_resolver import AdaptiveSalienceResolver
from runtime.calibrated_memory_resolver import CalibratedMemoryResolver

# Import ALFSR analysis modules
from analysis.decoder_entropy_monitor import DecoderEntropyMonitor
from analysis.steering_decay_controller import SteeringDecayController
from analysis.symbolic_continuation_stability_tracker import SymbolicContinuationStabilityTracker
from analysis.probabilistic_freedom_auditor import ProbabilisticFreedomAuditor

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_5"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# --------------------------------------------------------------------------
# Helper: chunked prefill
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# Helper: compute logit entropy
# --------------------------------------------------------------------------
def logit_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float().squeeze(0), dim=-1)
    log_p = torch.log(probs + 1e-12)
    return -(probs * log_p).sum().item()

# --------------------------------------------------------------------------
# Core single-run executor
# --------------------------------------------------------------------------
def execute_run(
    model, tokenizer, test_case, mode, resolver,
    max_new_tokens=64,
    temperature=0.7,
):
    """
    Execute one generation run.

    Returns dict with metrics + raw telemetry.
    """
    needle = test_case["needle"]
    input_ids = torch.tensor([test_case["tokens"]]).to("cuda")
    seq_len = input_ids.shape[1]

    # Trackers (created fresh every run)
    entropy_mon = DecoderEntropyMonitor(history_window=max_new_tokens)
    continuation_tracker = SymbolicContinuationStabilityTracker(needle)
    freedom_auditor = ProbabilisticFreedomAuditor()

    # Reset resolver state if ALFSR
    if hasattr(resolver, "reset_generation_state"):
        resolver.reset_generation_state()

    # --- Prefill ---
    past_key_values = prefill_chunked(model, input_ids, resolver)
    vram_after_prefill = torch.cuda.memory_allocated() / 1024**3

    # --- Decode ---
    curr_input = input_ids[:, -1:]
    curr_pos   = seq_len
    generated_tokens = []
    steering_trace   = []
    entropy_trace    = []
    freedom_trace    = []
    token_trace      = []

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

        # Scale by temperature
        scaled_logits = raw_logits / max(temperature, 1e-3)

        # Guide decoder (mode-specific)
        if resolver is not None and hasattr(resolver, "guide_decoder"):
            if isinstance(resolver, ALFSRResolver):
                guided_logits = resolver.guide_decoder(scaled_logits, attentions)
            elif isinstance(resolver, AttentionSteeringResolver):
                guided_logits = resolver.guide_decoder(scaled_logits, attentions)
            else:
                guided_logits = resolver.guide_decoder(scaled_logits)
        else:
            guided_logits = scaled_logits

        # Record entropy BEFORE sampling
        ent_snap = entropy_mon.record(guided_logits, generated_tokens[-1] if generated_tokens else -1)
        entropy_trace.append({
            "step": step,
            "entropy_nats": ent_snap.entropy_nats,
            "top1_prob": ent_snap.top1_prob,
            "top5_mass": ent_snap.top5_mass,
            "repetition_prob": ent_snap.repetition_prob,
        })

        # Sample
        probs = torch.softmax(guided_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_id = token.item()
        generated_tokens.append(token_id)

        # Record freedom
        freedom_snap = freedom_auditor.record(guided_logits, token_id)
        freedom_trace.append({
            "step": step,
            "entropy_nats": freedom_snap.entropy_nats,
            "classification": freedom_snap.classification,
            "is_looping": freedom_snap.is_looping,
        })

        # Record steering trace (ALFSR only)
        if isinstance(resolver, ALFSRResolver) and resolver.steering_trace:
            last_steer = resolver.steering_trace[-1]
            steering_trace.append(last_steer)

            # Notify ALFSR of the generated token
            resolver.record_generated_token(token_id, guided_logits)

        # Record token
        decoded_so_far = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        token_trace.append({
            "step": step,
            "token_id": token_id,
            "token_str": tokenizer.decode([token_id]),
            "generated_so_far": decoded_so_far,
        })

        # Continuation tracker (record every 4 steps to avoid overhead)
        if step % 4 == 0:
            continuation_tracker.record(generated_tokens, tokenizer)

        curr_input = token
        curr_pos += 1
        if token_id == tokenizer.eos_token_id:
            break

    duration = time.perf_counter() - start_gen
    tps = len(generated_tokens) / duration if duration > 0 else 0.0
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Exact match
    def normalize(t): return "".join(t.lower().split())
    exact_match = normalize(needle) in normalize(output_text)

    # Partial match (needle chars present)
    needle_chars = set(normalize(needle))
    output_chars = set(normalize(output_text))
    partial_match = len(needle_chars & output_chars) / max(len(needle_chars), 1)

    # Continuation summary
    cont_summary = continuation_tracker.summary()

    # Freedom summary
    freedom_summary = freedom_auditor.summary()

    return {
        "mode": mode,
        "ctx": input_ids.shape[1],
        "domain": test_case["domain"],
        "needle": needle,
        "output": output_text,
        "exact_match": exact_match,
        "partial_match": partial_match,
        "tps": tps,
        "ttft_ms": (start_gen - time.perf_counter()) * -1000,  # approximation
        "vram_gb": vram_after_prefill,
        "num_tokens_generated": len(generated_tokens),
        "continuation_summary": cont_summary,
        "freedom_summary": freedom_summary,
        "entropy_trace": entropy_trace,
        "freedom_trace": freedom_trace,
        "steering_trace": steering_trace,
        "token_trace": token_trace,
    }

# --------------------------------------------------------------------------
# Steering sweep executor
# --------------------------------------------------------------------------
STEERING_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0]

def run_steering_sweep(model, tokenizer, test_case):
    """
    Run the same test case across all steering strength levels.
    Returns a list of results (one per level).
    """
    sweep_results = []
    for strength in STEERING_LEVELS:
        resolver = ALFSRResolver(
            tokenizer,
            anchor_budget=test_case["ctx"] // 2,
            fidelity_budget=1024,
            initial_strength=strength,
            max_strength=strength,   # fix max = initial for sweep
        )
        result = execute_run(model, tokenizer, test_case, f"alfsr_sweep_{strength}", resolver)
        result["sweep_strength"] = strength
        sweep_results.append(result)
        print(f"  [SWEEP] strength={strength:.2f} | EM={result['exact_match']} | "
              f"TPS={result['tps']:.1f} | entropy_mean="
              f"{sum(e['entropy_nats'] for e in result['entropy_trace']) / max(len(result['entropy_trace']),1):.2f}")
    return sweep_results

# --------------------------------------------------------------------------
# Main validation runner
# --------------------------------------------------------------------------
class ValidationRunner20_5:
    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.suite     = MultidomainSymbolicSuite(tokenizer)
        self.noise_injector = NoisyContextInjector()

        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # Raw artifact files (opened once, appended throughout)
        self._open_files = {}
        for fname in [
            "raw_entropy_curves.jsonl",
            "raw_steering_decay.jsonl",
            "raw_probabilistic_freedom.jsonl",
            "raw_symbolic_continuations.jsonl",
            "raw_decoder_diversity.jsonl",
            "raw_retrieval_stability.jsonl",
            "raw_token_generation.jsonl",
        ]:
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
            self._open_files[fname] = open(path, "a")

        self.wallclock_log = open(os.path.join(RESULTS_DIR, "raw_wallclock_trace.log"), "w")
        self._log_wall(f"[START] Phase 20.5 ALFSR Validation @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _log_wall(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.wallclock_log.write(line + "\n")
        self.wallclock_log.flush()

    def _write(self, fname, obj):
        self._open_files[fname].write(json.dumps(obj) + "\n")
        self._open_files[fname].flush()

    def _make_resolver(self, mode, ctx_len):
        budget = ctx_len // 2
        if mode == "dense":
            return None
        elif mode == "sparse_baseline":
            return AdaptiveSalienceResolver(self.tokenizer, anchor_budget=budget, fidelity_budget=1024)
        elif mode == "dtascc_19_7":
            return CalibratedMemoryResolver(anchor_budget=budget, fidelity_budget=1024)
        elif mode == "aascsi_20_3":
            r = AttentionSteeringResolver(self.tokenizer, anchor_budget=budget, fidelity_budget=1024)
            r.logit_bias_strength = 15.0
            return r
        elif mode == "alfsr_20_5":
            return ALFSRResolver(self.tokenizer, anchor_budget=budget, fidelity_budget=1024,
                                 initial_strength=1.0, max_strength=8.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_single(self, mode, ctx_len, domain, use_noise=False):
        self._log_wall(f"Running mode={mode} ctx={ctx_len} domain={domain} noise={use_noise}")

        test_case = self.suite.create_domain_test_case(domain, ctx_len)
        test_case["ctx"] = ctx_len
        test_case["domain"] = domain

        if use_noise:
            noisy_prompt = self.noise_injector.inject_noise(test_case["full_prompt"], intensity=0.1)
            test_case["tokens"] = self.tokenizer.encode(noisy_prompt)

        resolver = self._make_resolver(mode, ctx_len)
        result = execute_run(self.model, self.tokenizer, test_case, mode, resolver)

        # --- Write raw artifacts ---
        for e in result["entropy_trace"]:
            e.update({"mode": mode, "ctx": ctx_len, "domain": domain})
            self._write("raw_entropy_curves.jsonl", e)

        for f in result["freedom_trace"]:
            f.update({"mode": mode, "ctx": ctx_len, "domain": domain})
            self._write("raw_probabilistic_freedom.jsonl", f)

        for s in result.get("steering_trace", []):
            s.update({"mode": mode, "ctx": ctx_len, "domain": domain})
            self._write("raw_steering_decay.jsonl", s)

        self._write("raw_symbolic_continuations.jsonl", {
            "mode": mode, "ctx": ctx_len, "domain": domain,
            **result["continuation_summary"],
        })

        self._write("raw_retrieval_stability.jsonl", {
            "mode": mode, "ctx": ctx_len, "domain": domain,
            "exact_match": result["exact_match"],
            "partial_match": result["partial_match"],
            "tps": result["tps"],
            "vram_gb": result["vram_gb"],
            "output": result["output"],
            "needle": result["needle"],
        })

        self._write("raw_decoder_diversity.jsonl", {
            "mode": mode, "ctx": ctx_len, "domain": domain,
            **result["freedom_summary"],
        })

        for t in result["token_trace"][:20]:   # first 20 tokens
            t.update({"mode": mode, "ctx": ctx_len, "domain": domain})
            self._write("raw_token_generation.jsonl", t)

        print(f"  EM={result['exact_match']} | Partial={result['partial_match']:.2f} | "
              f"TPS={result['tps']:.1f} | Freedom={result['freedom_summary']['probabilistic_fraction']:.2f}")
        return result

    def run_steering_sweep_suite(self, domain="activation_code", ctx_len=4096):
        self._log_wall(f"[SWEEP] domain={domain} ctx={ctx_len}")
        test_case = self.suite.create_domain_test_case(domain, ctx_len)
        test_case["ctx"] = ctx_len
        test_case["domain"] = domain
        sweep_results = run_steering_sweep(self.model, self.tokenizer, test_case)
        for r in sweep_results:
            self._write("raw_retrieval_stability.jsonl", {
                "mode": r["mode"],
                "sweep_strength": r["sweep_strength"],
                "ctx": ctx_len,
                "domain": domain,
                "exact_match": r["exact_match"],
                "partial_match": r["partial_match"],
                "tps": r["tps"],
                "vram_gb": r["vram_gb"],
                "output": r["output"],
                "needle": r["needle"],
            })
            for s in r.get("steering_trace", []):
                s.update({"sweep_strength": r["sweep_strength"]})
                self._write("raw_steering_decay.jsonl", s)
        return sweep_results

    def run_full_suite(self):
        modes   = ["dense", "sparse_baseline", "dtascc_19_7", "aascsi_20_3", "alfsr_20_5"]
        contexts = [4096, 8192]
        domains  = [
            "activation_code", "api_key", "json_snippet",
            "code_fragment", "multilingual",
        ]

        all_results = []
        for ctx in contexts:
            for domain in domains:
                for mode in modes:
                    try:
                        r = self.run_single(mode, ctx, domain)
                        all_results.append(r)
                    except Exception as ex:
                        self._log_wall(f"[ERROR] {mode}/{ctx}/{domain}: {ex}")

        return all_results

    def generate_reports(self, all_results):
        self._log_wall("[REPORTS] Generating markdown reports...")

        # Helper: group by mode
        def by_mode(results):
            grouped = {}
            for r in results:
                grouped.setdefault(r["mode"], []).append(r)
            return grouped

        grouped = by_mode(all_results)
        modes   = list(grouped.keys())

        # ---- 1. Low-Force Retrieval Report ----
        lines = ["# Phase 20.5: Low-Force Retrieval Performance\n\n"]
        lines.append("| Mode | Ctx | Domain | Exact Match | Partial Match | TPS | VRAM |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for r in all_results:
            lines.append(f"| {r['mode']} | {r['ctx']} | {r['domain']} | "
                         f"{'✅' if r['exact_match'] else '❌'} | "
                         f"{r['partial_match']:.2f} | {r['tps']:.1f} | {r['vram_gb']:.2f} GB |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_low_force_retrieval.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # ---- 2. Entropy Stability Report ----
        lines = ["# Phase 20.5: Entropy Stability\n\n"]
        lines.append("| Mode | Mean Entropy (nats) | Collapse Events | Probabilistic % |\n")
        lines.append("|---|---|---|---|\n")
        for mode, results in grouped.items():
            all_entropy = [e["entropy_nats"] for r in results for e in r.get("entropy_trace", [])]
            mean_ent = sum(all_entropy) / max(len(all_entropy), 1)
            collapse = sum(1 for e in all_entropy if e < 0.5)
            prob_frac = sum(r["freedom_summary"]["probabilistic_fraction"] for r in results) / max(len(results), 1)
            lines.append(f"| {mode} | {mean_ent:.3f} | {collapse} | {prob_frac:.1%} |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_entropy_stability.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # ---- 3. Probabilistic Freedom Report ----
        lines = ["# Phase 20.5: Probabilistic Freedom Audit\n\n"]
        lines.append("| Mode | Probabilistic % | Guided % | Forced % | Looping Detected |\n")
        lines.append("|---|---|---|---|---|\n")
        for mode, results in grouped.items():
            p = sum(r["freedom_summary"]["probabilistic_fraction"] for r in results) / max(len(results), 1)
            g = sum(r["freedom_summary"]["guided_fraction"] for r in results) / max(len(results), 1)
            fo = sum(r["freedom_summary"]["forced_fraction"] for r in results) / max(len(results), 1)
            loop = any(r["freedom_summary"]["looping_detected"] for r in results)
            lines.append(f"| {mode} | {p:.1%} | {g:.1%} | {fo:.1%} | {'YES' if loop else 'no'} |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_probabilistic_freedom.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # ---- 4. Decoder Legitimacy Report ----
        lines = ["# Phase 20.5: Decoder Legitimacy Assessment\n\n"]
        lines.append("Classifies retrieval as: **emergent** | **guided** | **forced**\n\n")
        lines.append("| Mode | Domain | Ctx | Classification | Exact Match |\n")
        lines.append("|---|---|---|---|---|\n")
        for r in all_results:
            p_frac = r["freedom_summary"]["probabilistic_fraction"]
            f_frac = r["freedom_summary"]["forced_fraction"]
            if f_frac > 0.5:
                cls = "**forced**"
            elif p_frac > 0.6:
                cls = "emergent"
            else:
                cls = "guided"
            lines.append(f"| {r['mode']} | {r['domain']} | {r['ctx']} | {cls} | "
                         f"{'✅' if r['exact_match'] else '❌'} |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_decoder_legitimacy.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # ---- 5. Compute Balance Report ----
        lines = ["# Phase 20.5: Compute Balance\n\n"]
        lines.append("| Mode | Mean TPS | Mean VRAM | Tokens/Run |\n")
        lines.append("|---|---|---|---|\n")
        for mode, results in grouped.items():
            mean_tps  = sum(r["tps"]  for r in results) / max(len(results), 1)
            mean_vram = sum(r["vram_gb"] for r in results) / max(len(results), 1)
            mean_toks = sum(r["num_tokens_generated"] for r in results) / max(len(results), 1)
            lines.append(f"| {mode} | {mean_tps:.1f} | {mean_vram:.2f} GB | {mean_toks:.0f} |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_compute_balance.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        # ---- 6. Failure Analysis Report ----
        lines = ["# Phase 20.5: Failure Analysis\n\n"]
        failures = [r for r in all_results if not r["exact_match"]]
        lines.append(f"Total failures: {len(failures)} / {len(all_results)}\n\n")
        lines.append("| Mode | Domain | Ctx | Partial Match | Output (truncated) | Expected |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for r in failures:
            out_trunc = r["output"][:60].replace("\n", " ")
            lines.append(f"| {r['mode']} | {r['domain']} | {r['ctx']} | "
                         f"{r['partial_match']:.2f} | {out_trunc}... | {r['needle'][:40]} |\n")
        with open(os.path.join(REPORTS_DIR, "reconstruction_20_5_failure_analysis.md"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        self._log_wall("[REPORTS] All 6 reports written.")

    def close(self):
        self._log_wall(f"[END] Phase 20.5 ALFSR Validation complete @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        for f in self._open_files.values():
            f.close()
        self.wallclock_log.close()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("[PHASE 20.5] ALFSR - Adaptive Low-Force Symbolic Recovery")
    print("[PHASE 20.5] Initializing hardware stack...")

    loader    = Qwen7BRealLoader()
    model     = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    runner = ValidationRunner20_5(model, tokenizer)

    try:
        # --- Steering sensitivity sweep (Phase 20.5 research question #1) ---
        runner._log_wall("[SWEEP] Steering sensitivity sweep: activation_code @ 4k")
        sweep_results = runner.run_steering_sweep_suite(domain="activation_code", ctx_len=4096)

        # --- Full mode × context × domain matrix ---
        runner._log_wall("[MATRIX] Full validation matrix...")
        all_results = runner.run_full_suite()

        # --- Generate reports ---
        runner.generate_reports(all_results)

    finally:
        runner.close()

    print("\n[PHASE 20.5] DONE. Artifacts in:", RESULTS_DIR)
    print("[PHASE 20.5] Reports in:", REPORTS_DIR)
