import os
import torch
from runtime.bic_resolver import BICResolver
from benchmark_component_registry import registry
from telemetry_scope_tracker import scope_tracker

def reset_bic():
    registry.participating = set()
    scope_tracker.scope = {k: False for k in scope_tracker.scope}

def test_subsystem_mode():
    print("\n--- Testing SUBSYSTEM Mode ---")
    reset_bic()
    resolver = BICResolver("SUBSYSTEM")
    
    # Simulate subsystem activity
    registry.register("triton_kernels")
    registry.register("kv_virtualization")
    scope_tracker.set_scope("kernels", True)
    
    resolver.finalize_benchmark("subsystem_benchmark_report.md")

def test_integrated_mode():
    print("\n--- Testing INTEGRATED Mode ---")
    reset_bic()
    resolver = BICResolver("INTEGRATED")
    
    # Simulate integrated activity
    registry.register("triton_kernels")
    registry.register("kv_virtualization")
    registry.register("logits")
    registry.register("mlp")
    scope_tracker.set_scope("kernels", True)
    scope_tracker.set_scope("model_weights", True)
    scope_tracker.set_scope("wall_clock", True)
    
    resolver.finalize_benchmark("integrated_benchmark_report.md")

def test_production_mode():
    print("\n--- Testing PRODUCTION Mode ---")
    reset_bic()
    resolver = BICResolver("PRODUCTION")
    
    # Simulate production activity
    registry.register("embeddings")
    registry.register("tokenizer")
    registry.register("logits")
    registry.register("mlp")
    registry.register("sampling")
    registry.register("triton_kernels")
    registry.register("kv_virtualization")
    registry.register("batching")
    registry.register("concurrency")
    
    scope_tracker.set_scope("kernels", True)
    scope_tracker.set_scope("model_weights", True)
    scope_tracker.set_scope("wall_clock", True)
    scope_tracker.set_scope("gpu_allocations", True)
    
    resolver.finalize_benchmark("production_benchmark_report.md")

def test_failure_case():
    print("\n--- Testing FAILURE Case (Production mode missing components) ---")
    reset_bic()
    resolver = BICResolver("PRODUCTION")
    
    # Missing tokenizer/sampling
    registry.register("logits")
    
    resolver.finalize_benchmark("failed_benchmark_report.md")

if __name__ == "__main__":
    test_subsystem_mode()
    test_integrated_mode()
    test_production_mode()
    test_failure_case()
    
    print("\n[BIC] Validation suite complete.")
