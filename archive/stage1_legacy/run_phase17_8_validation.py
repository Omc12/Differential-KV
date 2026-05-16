import time
import torch
import os
import json
import numpy as np
from runtime.quantized_sparse_runtime import QuantizedSparseRuntime
from runtime.real_kv_pressure_manager import RealKVPressureManager

def run_native_validation(model_scale, context_len):
    print(f"\n[TEST] Model: {model_scale}, Context: {context_len}")
    
    runtime = QuantizedSparseRuntime(model_scale=model_scale, sparse_budget=0.1)
    pressure_manager = RealKVPressureManager(runtime.tokenizer)
    
    prompt = pressure_manager.generate_long_prompt(context_len)
    
    # We use a small number of tokens but wall-clock timing is real.
    res = runtime.generate(prompt, max_new_tokens=20, use_sparse=True)
    
    print(f"TPS [MEASURED]: {res['tps']:.2f}")
    print(f"VRAM [MEASURED]: {res['vram_gb']:.2f} GB")
    print(f"Kernel Launches: {res['kernels']}")
    
    return {
        "model_scale": model_scale,
        "context_len": context_len,
        "tps": res["tps"],
        "vram_gb": res["vram_gb"],
        "kernels": res["kernels"],
        "stats": runtime.get_stats()
    }

def run_phase17_8_validation():
    print("=== Phase 17.8: CUDA/TRITON ACCELERATION & ASYNCHRONOUS STREAMING ===")
    
    results_dir = "results/reconstruction_17_8"
    os.makedirs(results_dir, exist_ok=True)
    
    test_matrix = [
        ("7B", 16000),
        ("13B", 4000),
        ("32B", 4000)
    ]
    
    full_results = []
    
    for model_scale, context_len in test_matrix:
        try:
            res = run_native_validation(model_scale, context_len)
            full_results.append(res)
        except Exception as e:
            print(f"[ERROR] Run failed: {e}")

    # Generate Reports
    generate_reports(results_dir, full_results)

def generate_reports(results_dir, full_results):
    # Report 1: GPU-Native TPS
    with open(f"{results_dir}/reconstruction_17_8_true_tps.md", "w") as f:
        f.write("# Phase 17.8 GPU-Native TPS Report\n\n")
        f.write("## [MEASURED] Accelerated Throughput\n")
        f.write("| Model | Context | Previous TPS (17.7) | Accelerated TPS (17.8) | Speedup |\n")
        f.write("|---|---|---|---|---|\n")
        
        # Previous 17.7 values
        prev = {("7B", 16000): 0.39, ("13B", 4000): 0.97, ("32B", 4000): 0.15}
        
        for res in full_results:
            p_tps = prev.get((res["model_scale"], res["context_len"]), 0)
            speedup = (res["tps"] / p_tps) if p_tps > 0 else 0
            f.write(f"| {res['model_scale']} | {res['context_len']//1000}k | {p_tps:.2f} | {res['tps']:.2f} | {speedup:.1f}x |\n")

    # Report 2: GPU Overlap & Kernels
    with open(f"{results_dir}/reconstruction_17_8_gpu_overlap.md", "w") as f:
        f.write("# Phase 17.8 GPU Overlap & Kernel Report\n\n")
        f.write("## [MEASURED] Kernel Metrics\n")
        for res in full_results:
            f.write(f"### {res['model_scale']} Scale\n")
            f.write(f"- Total Kernels: {res['kernels']}\n")
            f.write(f"- CUDA Stream Overlap: {res['stats']['kernel_stats']['overlap']['stream_overlap_efficiency']*100:.1f}%\n")
            f.write(f"- TPS: {res['tps']:.2f}\n\n")

if __name__ == "__main__":
    run_phase17_8_validation()
