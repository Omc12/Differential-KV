import asyncio
import torch
import logging
import time
from typing import Dict, List, Any
from runtime.dco_resolver import DCOResolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DCOValidation")

async def run_dco_validation():
    logger.info("Starting Phase 25.1 — DCO Validation...")
    
    # 1. Initialize DCO Infrastructure
    devices = ["cuda:0", "cuda:1"]
    resolver = DCOResolver(devices)
    
    # 2. Setup Mock Fabric
    segments = [f"seg_{i}" for i in range(20)]
    tensors = {sid: torch.randn(1, 128) for sid in segments}
    
    async def mock_fabric_access(sid: str):
        # Simulate some delay
        await asyncio.sleep(0.02)
        return tensors[sid]

    logger.info("--- Test 1: Optimized Remote Access (Overlap & Compression) ---")
    # Perform several accesses to build history and test overlap
    for i in range(10):
        sid = segments[i]
        for _ in range(5):
            await resolver.optimized_remote_access(sid, "cuda:1", mock_fabric_access)
        logger.info(f"Accessed {sid} with DCO optimization (5x).")

    logger.info("--- Test 2: Predictive Prefetching ---")
    # Access segments in a pattern to trigger prefetch
    # Pattern: 0->1, 1->2, ...
    for i in range(5):
        await resolver.optimized_remote_access(segments[i], "cuda:0", mock_fabric_access)
    
    # Predict next after 'seg_4'
    prediction = resolver.prefetch_predictor.predict_next("seg_4")
    logger.info(f"Prediction after seg_4: {prediction}")

    logger.info("--- Test 3: Locality-Aware Mapping ---")
    # Access seg_10 repeatedly from cuda:1
    for _ in range(10):
        await resolver.optimized_remote_access("seg_10", "cuda:1", mock_fabric_access)
    
    optimal_dev = resolver.device_mapper.get_optimal_device("seg_10")
    logger.info(f"Optimal device for seg_10: {optimal_dev}")

    # 3. Collect Metrics
    metrics = resolver.get_dco_metrics()
    
    logger.info("\n=== DCO Phase 25.1 Metrics ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    
    # 4. Final Validation Check
    success = (
        metrics["cross_device_latency_reduction"] >= 0.25 and
        metrics["remote_kv_bandwidth_reduction"] >= 0.30 and
        metrics["transfer_overlap_efficiency"] >= 0.80 and
        metrics["migration_thrash_risk"] < 0.2 and
        metrics["distributed_symbolic_continuity"] == 1.0 and
        metrics["scheduler_stability"] == "stable"
    )
    
    if success:
        logger.info("\nPHASE 25.1 DCO VALIDATION SUCCESSFUL")
    else:
        logger.error("\nPHASE 25.1 DCO VALIDATION FAILED")

if __name__ == "__main__":
    asyncio.run(run_dco_validation())
