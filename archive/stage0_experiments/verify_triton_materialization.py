import argparse
import torch
import json
import os
from enable_execution_audit import auditor, patch_runtime

def verify_triton(args):
    print(f"Verifying Triton Materialization: {args.model}")
    
    patch_runtime()
    auditor.configure(
        trace_triton=True,
        trace_kernels=True,
        trace_fallbacks=True,
        export_trace="telemetry/triton_verification_trace.json"
    )

    # 1. Test standard low-rank reconstruction
    print("Testing Low-Rank Triton Kernel...")
    U = torch.randn(32, 16, device=args.device, dtype=torch.float16)
    V = torch.randn(16, 128, device=args.device, dtype=torch.float16)
    anchor = torch.randn(128, device=args.device, dtype=torch.float16)
    
    # This should trigger the audit log via the patch
    from runtime.triton_diffkv import TritonDiffKV
    res = TritonDiffKV.reconstruct_lowrank(U, V, anchor)
    
    # 2. Test sparse reconstruction
    print("Testing Sparse Triton Path...")
    sparse_indices = torch.tensor([0, 10, 20], device=args.device)
    sparse_values = torch.tensor([1.0, 2.0, 3.0], device=args.device, dtype=torch.float16)
    res_sparse = TritonDiffKV.reconstruct_lowrank_sparse(U, V, anchor, sparse_indices, sparse_values)

    # Analyze logs
    triton_launches = [e for e in auditor.trace_log if e["category"] == "triton" and e["event_type"] == "launch"]
    kernel_successes = [e for e in auditor.trace_log if e["category"] == "kernels" and e["event_type"] == "custom_kernel_success"]
    fallbacks = [e for e in auditor.trace_log if e["category"] == "fallbacks"]

    report = {
        "triton_launches": len(triton_launches),
        "kernel_successes": len(kernel_successes),
        "fallback_events": len(fallbacks),
        "active_kernels": list(set([e["data"]["kernel"] for e in triton_launches])),
        "fallback_details": [e["data"] for e in fallbacks]
    }

    print("\n--- Triton Materialization Report ---")
    print(f"Triton Launches: {report['triton_launches']}")
    print(f"Kernel Successes: {report['kernel_successes']}")
    print(f"Fallbacks: {report['fallback_events']}")
    
    if report['triton_launches'] > 0 and report['fallback_events'] == 0:
        print("STATUS: Triton path is FULLY ACTIVE and MATERIALIZED.")
    elif report['triton_launches'] > 0 and report['fallback_events'] > 0:
        print("STATUS: Triton path is ACTIVE but experiencing FALLBACKS.")
    else:
        print("STATUS: Triton path is INACTIVE or failing entirely.")

    if args.export:
        os.makedirs(os.path.dirname(args.export), exist_ok=True)
        with open(args.export, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report exported to {args.export}")

    auditor.export()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dump-active-kernels", action="store_true")
    parser.add_argument("--dump-triton-launches", action="store_true")
    parser.add_argument("--dump-fallback-events", action="store_true")
    parser.add_argument("--export", type=str, default="telemetry/triton_materialization_report.json")
    
    args = parser.parse_args()
    verify_triton(args)
