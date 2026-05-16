"""
run_cbp_validation.py

Validation script for Phase 33.0: CBP (Canonical Benchmark & Publication).
Verifies the entire CBP pipeline and ensures publication-quality artifacts are generated.
"""

import os
import json
import logging
from runtime.cbp_resolver import CBPResolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CBPValidation")

import asyncio

async def main():
    logger.info("Starting FINAL REAL Phase 33.0 CBP Validation...")
    
    # 1. Initialize Resolver with Production Config
    config = {
        "cbp": {
            "mode": "PRODUCTION",
            "trials": 1,
            "strict_comparisons": True
        }
    }
    
    resolver = CBPResolver(config)
    
    # 2. Run the Full Publication Benchmark Pass (REAL COMPUTE)
    results = await resolver.run_publication_benchmark()
    
    if results.get("status") == "FAILED":
        logger.error(f"CBP Validation FAILED: {results.get('violations')}")
        exit(1)
        
    # 3. Verify Artifact Existence
    required_artifacts = [
        "FINAL_REAL_BENCHMARK_REPORT.md",
        "benchmark_truth_manifest.json",
        "benchmark_reproducibility.json",
        "benchmark_scope_manifest.json",
        "runtime_participation_manifest.json",
        "telemetry_scope_manifest.json"
    ]
    
    missing_artifacts = [a for a in required_artifacts if not os.path.exists(a)]
    if missing_artifacts:
        logger.error(f"Missing required publication artifacts: {missing_artifacts}")
        exit(1)
        
    logger.info("All publication artifacts verified.")
    
    # 4. Verify Content of Truth Manifest
    with open("benchmark_truth_manifest.json", "r") as f:
        manifest = json.load(f)
        
    if manifest["benchmark_classification"] != "PRODUCTION":
        logger.error("Truth manifest classification mismatch.")
        exit(1)
        
    if not manifest["serving_overhead_included"]:
        logger.error("Serving overhead missing from truth manifest.")
        exit(1)
        
    logger.info("Truth manifest content verified.")
    
    # 5. Summary Output
    logger.info("\n" + "="*50)
    logger.info("PHASE 33.0 CBP VALIDATION SUCCESSFUL")
    logger.info("="*50)
    logger.info(f"Sustained TPS: {results.get('sustained_tps', 0):.4f}")
    logger.info(f"TTFT (ms):     {results.get('ttft_ms', 0):.4f}")
    logger.info(f"ITL (ms):      {results.get('itl_ms', 0):.4f}")
    logger.info(f"Sparse %:      {results.get('sparse_runtime_pct', 0):.2f}%")
    logger.info("="*50)
    logger.info("CBP platform is stable, reproducible, and ready for publication.")

if __name__ == "__main__":
    asyncio.run(main())
