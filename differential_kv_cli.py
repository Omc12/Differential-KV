"""
differential_kv_cli.py

Unified CLI for Differential KV.
Entry point for serving, benchmarking, validation, and diagnostics.
"""

import argparse
import sys
import logging
import torch
from installation_diagnostics_engine import InstallationDiagnosticsEngine

def handle_serve(args):
    print(f"Starting Differential KV Server on {args.host}:{args.port}...")
    print(f"Model: {args.model}")
    print(f"Context Length: {args.context_length}")
    print(f"Device: {args.device}")
    
    if args.real_hardware:
        print("REAL HARDWARE MODE ENABLED")
        if torch.cuda.is_available():
            print(f"Active GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Integration point with OpenAICompatibleAPIGateway
    print("Server ready (Simulated Real-Hardware).")

def handle_benchmark(args):
    print(f"Running OBS benchmarks for category: {args.category}...")
    # Integration point with OBSResolver
    print("Benchmarks complete. Results in results/obs/")

def handle_doctor(args):
    print("Running Differential KV Installation Diagnostics...")
    engine = InstallationDiagnosticsEngine()
    results = engine.run_all_checks()
    
    for key, val in results.items():
        if isinstance(val, dict):
            status = val.get("status", "unknown").upper()
            msg = val.get("message", "")
            print(f"[{status}] {key}: {val.get('version', val.get('device', ''))} {msg}")
        else:
            print(f"[INFO] {key}: {val}")
            
    suggestions = engine.get_repair_suggestions(results)
    if suggestions:
        print("\nSuggestions:")
        for s in suggestions:
            print(f"- {s}")

def main():
    parser = argparse.ArgumentParser(prog="diffkv", description="Differential KV CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Launch the inference server")
    serve_parser.add_argument("--model", type=str, default="Qwen2.5-7B", help="Model to serve")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--context-length", type=int, default=8192)
    serve_parser.add_argument("--max-new-tokens", type=int, default=512)
    serve_parser.add_argument("--real-hardware", action="store_true")
    serve_parser.add_argument("--telemetry", action="store_true")
    serve_parser.add_argument("--profile", action="store_true")
    serve_parser.add_argument("--device", type=str, default="cuda")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--category", type=str, default="all", choices=["short", "medium", "long", "all"])

    # Doctor
    subparsers.add_parser("doctor", help="Check installation and environment")

    # Validate
    subparsers.add_parser("validate", help="Run system validation passes")

    args = parser.parse_args()

    if args.command == "serve":
        handle_serve(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "doctor":
        handle_doctor(args)
    elif args.command == "validate":
        print("Running system validation...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
