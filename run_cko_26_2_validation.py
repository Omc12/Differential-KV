import torch
import logging
import time
from typing import Dict, List, Any
from runtime.cko_resolver import CKOResolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CKOValidation")

def run_cko_validation():
    logger.info("Starting Phase 26.2 — CKO Validation...")
    
    # 1. Initialize CKO Infrastructure
    devices = ["cuda:0"]
    resolver = CKOResolver(devices)
    resolver.initialize_native_runtime()
    
    # 2. Test Optimized Decode Path
    x = torch.randn(1, 1, 128)
    logger.info("--- Test 1: Shared-Memory Staging & Warp Scheduling ---")
    for i in range(10):
        # Access segments to build cache history
        seg_id = f"seg_{i % 5}"
        resolver.optimized_decode_step(x, seg_id)
    
    logger.info("Completed 10 optimized decode steps.")

    logger.info("--- Test 2: CUDA Graph Capture & Replay ---")
    def dummy_op(tensor): return tensor * 2
    resolver.graph_engine.capture_graph("decode_graph", dummy_op, x)
    for _ in range(5):
        resolver.graph_engine.replay_graph("decode_graph")
    
    logger.info("Captured and replayed CUDA Graph 5 times.")

    logger.info("--- Test 3: Persistent Kernel Stability ---")
    resolver.persistent_kernel.stop_kernel()
    metrics = resolver.get_cko_metrics()
    logger.info(f"Persistent Kernel Uptime: {metrics['persistent_decode_uptime']:.4f}s")

    # 3. Collect Metrics
    logger.info("\n=== CKO Phase 26.2 Metrics ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    
    # 4. Final Validation Check
    success = (
        metrics["cuda_kernel_integrity"] == 1.0 and
        metrics["cuda_graph_replay_stability"] == 1.0 and
        metrics["shared_memory_hit_rate"] >= 0.5 and # With 10 accesses on 5 segs
        metrics["warp_scheduling_efficiency"] > 0.0 and
        metrics["persistent_decode_uptime"] > 0.0
    )
    
    if success:
        logger.info("\nPHASE 26.2 CKO VALIDATION SUCCESSFUL")
    else:
        logger.error("\nPHASE 26.2 CKO VALIDATION FAILED")

if __name__ == "__main__":
    run_cko_validation()
