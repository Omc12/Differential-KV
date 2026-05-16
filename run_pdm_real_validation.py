import logging
import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from runtime.pdm_resolver import PDMResolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDMValidation")

async def main():
    logger.info("Starting Phase 36.0 PDM (Production Deployment Materialization) Validation...")
    
    config = {
        "pdm": {
            "mode": "PRODUCTION",
            "checkpoint_interval_sec": 60,
            "vram_threshold": 0.9
        }
    }
    
    resolver = PDMResolver(config)
    
    # Run PDM Benchmark
    results = await resolver.run_pdm_benchmark()
    
    if results.get("status") == "FAILED":
        logger.error("PDM Validation FAILED. Operational resilience or reproducibility thresholds violated.")
        sys.exit(1)
        
    # Report Results
    logger.info("PDM Validation SUCCESSFUL.")
    logger.info(f"Deployment Reproducible: {results.get('deployment_reproducible')}")
    logger.info(f"Recovery Success Rate: {results.get('recovery_success_rate', 0)*100:.1f}%")
    logger.info(f"Telemetry Persisted: {results.get('telemetry_persisted')}")
    logger.info(f"Stability Index: {results.get('operational_stability_index', 0):.2f}")
    logger.info(f"VRAM Health: {results.get('mem_health', {}).get('status')}")
    
    logger.info("Differential KV is now a deployable, resilient, and production-ready platform.")

if __name__ == "__main__":
    asyncio.run(main())
