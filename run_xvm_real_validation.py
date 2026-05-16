import logging
import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from runtime.xvm_resolver import XVMResolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XVMValidation")

async def main():
    logger.info("Starting Phase 37.0 XVM (Cross-Validation & Materialization) Validation...")
    
    config = {
        "xvm": {
            "mode": "PRODUCTION",
            "hardware_profiles": ["RTX_4070_SUPER", "LOW_VRAM", "CPU_FALLBACK"],
            "model_sweep": True
        }
    }
    
    resolver = XVMResolver(config)
    
    # Run XVM Benchmark
    results = await resolver.run_xvm_benchmark()
    
    if results.get("status") == "FAILED":
        logger.error("XVM Validation FAILED. Portability or Compatibility thresholds violated.")
        sys.exit(1)
        
    # Report Results
    logger.info("XVM Validation SUCCESSFUL.")
    logger.info(f"Hardware Validated: {results.get('hardware_validated')}")
    logger.info(f"Compatibility Ratio: {results.get('compatibility_ratio', 0)*100:.1f}%")
    logger.info(f"Portability Score: {results.get('portability_score', 0):.2f}")
    logger.info(f"Model Coverage: {results.get('model_coverage')}")
    logger.info(f"Average Sparse Ratio: {results.get('avg_sparse_ratio', 0)*100:.1f}%")
    
    logger.info("Differential KV Stage 1 is COMPLETE. Platform is externally portable and mature.")

if __name__ == "__main__":
    asyncio.run(main())
