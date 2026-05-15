import torch
import logging
import asyncio
from typing import Dict, List, Any
from runtime.cko_resolver import CKOResolver
from runtime.nko_resolver import NKOResolver
from runtime.cko_nko_integration_patch import CKONKOIntegrationPatch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NKOValidation")

async def run_nko_validation():
    logger.info("Starting Phase 26.1 & 26.2a — NKO/CKO Validation...")
    
    # 1. Initialize Combined Infrastructure
    devices = ["cuda:0", "cuda:1"]
    cko = CKOResolver(devices)
    cko.initialize_native_runtime()
    
    nko = NKOResolver(devices, cko)
    patch = CKONKOIntegrationPatch(cko, nko)
    
    # 2. Apply Integration Patch (Phase 26.2a)
    patch.apply_patch_stabilization()
    
    # 3. Test Distributed CUDA-Native Execution
    logger.info("--- Test 1: Distributed Persistent Decode & NCCL Sync ---")
    nko.execute_distributed_nccl_step("dist_task_0", "seg_shared_0")
    
    logger.info("--- Test 2: P2P SRAM-to-SRAM Transport ---")
    # Verify SMEM staging hit rate after P2P
    metrics = nko.get_nko_metrics()
    logger.info(f"P2P-SMEM Hit Rate: {metrics['p2p_smem_hit_rate']}")

    logger.info("--- Test 3: NCCL Graph Stability ---")
    logger.info(f"NCCL Graph Stability: {metrics['nccl_graph_stability']}")

    # 4. Collect Final Metrics
    logger.info("\n=== NKO Phase 26.1 & 26.2a Metrics ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    
    patch_metrics = patch.get_patch_metrics()
    for k, v in patch_metrics.items():
        logger.info(f"{k}: {v}")
    
    # 5. Final Validation Check
    success = (
        metrics["nccl_graph_stability"] == 1.0 and
        metrics["distributed_persistent_uptime"] == 1.0 and
        metrics["distributed_replay_accuracy"] == 1.0 and
        patch_metrics["integration_stability_index"] == 1.0
    )
    
    if success:
        logger.info("\nPHASE 26.1 & 26.2a VALIDATION SUCCESSFUL")
    else:
        logger.error("\nPHASE 26.1 & 26.2a VALIDATION FAILED")

if __name__ == "__main__":
    asyncio.run(run_nko_validation())
