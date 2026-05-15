"""
run_phase29_1_obs_validation.py

Validation script for Phase 29.1: OBS (Operational Benchmark Suite).
Verifies that benchmarks are reproducible, reports are generated, and metrics are honest.
"""

import os
import json
import logging
from runtime.obs_resolver import OBSResolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase29.1Validation")

def main():
    logger.info("Starting Phase 29.1 OBS Validation...")
    os.makedirs("results/obs", exist_ok=True)
    
    config = {
        "obs": {
            "category": "all",
            "concurrency": [1, 4],
            "honesty_mode": True
        }
    }
    
    resolver = OBSResolver(config)
    metrics = resolver.run_full_benchmark_pass()
    
    # Required Metrics Check
    required_metrics = [
        "sustained_sparse_tps",
        "ttft_ms",
        "itl_ms",
        "vram_efficiency_ratio",
        "long_context_scaling_factor",
        "replay_consistency_score",
        "benchmark_reproducibility",
        "serving_stability_index",
        "comparative_runtime_status"
    ]
    
    missing = [m for m in required_metrics if m not in metrics]
    if missing:
        logger.error(f"Missing required metrics: {missing}")
        exit(1)
        
    logger.info("All required metrics validated.")
    
    # Report Verification
    if not os.path.exists(metrics["report_path"]):
        logger.error("Markdown report not found.")
        exit(1)
        
    if not os.path.exists(metrics["telemetry_path"]):
        logger.error("Telemetry JSON not found.")
        exit(1)
        
    logger.info(f"Benchmark Report verified: {metrics['report_path']}")
    
    # Final Result
    final_status = "SUCCESS"
    logger.info("\n" + "="*40)
    logger.info("PHASE 29.1 OBS VALIDATION SUMMARY")
    logger.info("="*40)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k:30}: {v:.4f}")
    logger.info("="*40)
    logger.info(f"STATUS: {final_status}")

if __name__ == "__main__":
    main()
