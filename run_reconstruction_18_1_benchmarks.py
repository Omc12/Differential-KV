import torch
import time
import json
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from models.tokenizer_consistency_lock import TokenizerConsistencyLock
from validation.token_generation_auditor import TokenGenerationAuditor
from validation.prompt_integrity_checker import PromptIntegrityChecker
from telemetry.strict_metric_taxonomy import StrictMetricTaxonomy
from telemetry.wallclock_enforcer import WallclockEnforcer
from telemetry.cuda_trace_correlator import CUDATraceCorrelator
from telemetry.token_timestamp_auditor import TokenTimestampAuditor
from analysis.bottleneck_mapper import BottleneckMapper
from analysis.failure_boundary_exporter import FailureBoundaryExporter

def run_grounded_benchmarks():
    print("="*60)
    print("PHASE 18.1 — REAL-MODEL SCIENTIFIC EXECUTION")
    print("="*60)

    # 1. Setup Grounded Telemetry
    taxonomy = StrictMetricTaxonomy()
    enforcer = WallclockEnforcer()
    cuda_tracer = CUDATraceCorrelator()
    token_auditor = TokenTimestampAuditor()
    bottleneck_mapper = BottleneckMapper()
    failure_exporter = FailureBoundaryExporter()
    prompt_checker = PromptIntegrityChecker()

    # 2. Load Real Model (NF4)
    loader = Qwen7BRealLoader()
    model = loader.load()
    
    tokenizer_lock = TokenizerConsistencyLock()
    _, tok_manifest = tokenizer_lock.verify()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    auditor = TokenGenerationAuditor(tokenizer)

    # 3. Test Matrix (4k, 8k, 16k)
    contexts = [4096, 8192, 16384]
    results = []

    for ctx_len in contexts:
        print(f"\n[RUN] Context: {ctx_len}")
        
        # Generate Grounded Prompt
        base_prompt = "Summarize the following architecture: Differential KV uses a hierarchical sparse memory engine to scale long-context serving on consumer hardware."
        p_hash = prompt_checker.hash_prompt(f"ctx_{ctx_len}", base_prompt)
        
        # Prepare input (repeat base to fill context)
        encoded_base = tokenizer.encode(base_prompt, return_tensors="pt")
        num_repeats = max(1, ctx_len // encoded_base.shape[1])
        input_ids = encoded_base.repeat(1, num_repeats)[:, :ctx_len].to("cuda")

        try:
            # ACTUAL CUDA EXECUTION
            enforcer.start()
            cuda_tracer.record_allocation(ctx_len)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False, # Deterministic
                    use_cache=True
                )
            
            duration = enforcer.stop()
            cuda_tracer.record_allocation(ctx_len + 32)
            
            # Audit Results
            trace = auditor.audit_generation(input_ids, output, 0, duration) # Auditor handles internal timing
            
            print(f"[RESULT] {taxonomy.log_measured('TPS', trace['tps'])}")
            print(f"[RESULT] {taxonomy.log_measured('VRAM', torch.cuda.memory_allocated() / (1024**3), 'GB')}")
            
            results.append({
                "context": ctx_len,
                "tps": trace['tps'],
                "vram_gb": torch.cuda.memory_allocated() / (1024**3),
                "status": "SUCCESS"
            })
            
        except Exception as e:
            print(f"[FAILURE] Context {ctx_len} failed: {e}")
            failure_exporter.record_failure(ctx_len, 1, str(e))
            bottleneck_mapper.record_bottleneck(f"Context_{ctx_len}", "OOM / Timeout", str(e))
            results.append({
                "context": ctx_len,
                "status": "FAILED",
                "error": str(e)
            })
            # Clear cache for next attempt
            torch.cuda.empty_cache()

    # 4. Final Reports
    bottleneck_mapper.export_report()
    generate_final_reports(results)

def generate_final_reports(results):
    report_path = "results/reconstruction_18_1/reconstruction_18_1_true_tps.md"
    with open(report_path, 'w') as f:
        f.write("# PHASE 18.1 — TRUE TPS BASELINE\n\n")
        f.write("| Context | Status | TPS [MEASURED] | VRAM [MEASURED] |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            tps = f"{r.get('tps', 0):.2f}" if r['status'] == "SUCCESS" else "N/A"
            vram = f"{r.get('vram_gb', 0):.2f} GB" if r['status'] == "SUCCESS" else "OOM"
            f.write(f"| {r['context']} | {r['status']} | {tps} | {vram} |\n")

if __name__ == "__main__":
    run_grounded_benchmarks()
