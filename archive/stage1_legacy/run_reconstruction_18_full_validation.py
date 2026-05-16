import torch
import time
import json
import os
import sys
from models.qwen_runtime_loader import QwenRuntimeLoader
from models.real_tokenizer_pipeline import RealTokenizerPipeline
from benchmarks.fairness_lock import FairnessLock
from benchmarks.context_integrity_checker import ContextIntegrityChecker
from validation.real_system_profiler import RealSystemProfiler
from validation.token_trace_archiver import TokenTraceArchiver
from validation.execution_taxonomy_enforcer import ExecutionTaxonomyEnforcer
from repro.environment_snapshot import EnvironmentSnapshot

def run_benchmarks():
    print("="*60)
    print("PHASE 18 — REAL-MODEL REPRODUCIBILITY & SCIENTIFIC BASELINING")
    print("="*60)

    # 1. Environment Snapshot
    snapshot_mgr = EnvironmentSnapshot()
    env_snapshot, snapshot_path = snapshot_mgr.capture()
    print(f"[INFO] Environment captured: {snapshot_path}")

    # 2. Initialization
    loader = QwenRuntimeLoader()
    tokenizer = RealTokenizerPipeline()
    fairness = FairnessLock()
    integrity = ContextIntegrityChecker()
    profiler = RealSystemProfiler()
    trace_archiver = TokenTraceArchiver("phase18_primary")
    taxonomy = ExecutionTaxonomyEnforcer()

    # 3. Model Loading
    print("[INFO] Loading Qwen2.5-7B-Instruct (NF4 Quantized)...")
    model = loader.load(use_flash_attn=False)
    
    # 4. Workloads
    contexts = [4096]
    concurrencies = [1, 2]
    
    results = []

    for ctx_len in contexts:
        for num_users in concurrencies:
            print(f"\n[RUN] Context: {ctx_len}, Users: {num_users}")
            
            # Prepare dummy prompt for ctx_len (real tokenization)
            # We use a simple prompt repeated to fill context for stress test
            base_prompt = "Explain the significance of the Differential KV architecture in long-context serving."
            encoded_base = tokenizer.encode(base_prompt)
            num_repeats = max(1, ctx_len // encoded_base.shape[1])
            full_prompt_ids = encoded_base.repeat(1, num_repeats)[:, :ctx_len]
            
            # Verify integrity
            is_valid, actual_len = integrity.check_sequence_length(full_prompt_ids)
            if not is_valid:
                print(f"[SKIP] {actual_len}")
                continue

            # Multi-user simulation
            total_tokens = 0
            total_time = 0
            
            for user in range(num_users):
                # Start Profiling
                profiler.start_timing()
                
                # Real Token Generation
                with torch.no_grad():
                    output = model.generate(
                        full_prompt_ids.to("cuda"),
                        max_new_tokens=32, # Faster for validation
                        do_sample=False,
                        use_cache=True
                    )
                
                elapsed = profiler.stop_timing()
                new_tokens = output.shape[1] - full_prompt_ids.shape[1]
                
                total_tokens += new_tokens
                total_time += elapsed
                
                # Record Trace
                trace_archiver.record_trace(
                    f"User {user} @ {ctx_len}",
                    tokenizer.decode(output[0][-new_tokens:]),
                    output[0][-new_tokens:].tolist(),
                    elapsed
                )

            avg_tps = total_tokens / total_time if total_time > 0 else 0
            vram = profiler.get_vram_usage()
            
            metric_line = f"TPS: {avg_tps:.2f}, VRAM: {vram:.2f}GB"
            labeled_metric = taxonomy.label_measured(metric_line)
            print(f"[RESULT] {labeled_metric}")
            
            results.append({
                "context": ctx_len,
                "users": num_users,
                "tps": avg_tps,
                "vram_gb": vram,
                "status": "SUCCESS"
            })

    # 5. Export Results
    trace_path = trace_archiver.archive()
    print(f"\n[INFO] Token traces archived to {trace_path}")
    
    with open("results/reconstruction_18/bench_results.json", 'w') as f:
        json.dump(results, f, indent=4)

    # 6. Generate Reports
    generate_reports(results, env_snapshot)
    
    # 7. Export Bundle (Phase 18F)
    from repro.run_bundle_export import export_all
    export_all()

def generate_reports(results, env):
    report_path = "results/reconstruction_18/reconstruction_18_real_model_tps.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# PHASE 18 — REAL-MODEL TPS REPORT\n\n")
        f.write(f"**Hardware**: {env['gpu']} ({env['gpu_vram_total_gb']:.2f} GB)\n")
        f.write(f"**Model**: Qwen2.5-7B-Instruct (4-bit NF4)\n")
        f.write(f"**Runtime**: Differential KV (Transformers Backend)\n\n")
        
        f.write("| Context | Users | TPS [MEASURED] | VRAM [MEASURED] |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['context']} | {r['users']} | {r['tps']:.2f} | {r['vram_gb']:.2f} GB |\n")
            
    print(f"[INFO] Report generated: {report_path}")

if __name__ == "__main__":
    try:
        run_benchmarks()
    except Exception as e:
        print(f"[FATAL] Phase 18 Validation failed: {e}")
        sys.exit(1)
