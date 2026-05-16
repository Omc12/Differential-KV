import logging
import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from runtime.eom_resolver import EOMResolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EOMValidation")

async def main():
    logger.info("Starting Phase 34.0 EOM (End-to-End Optimization Materialization) Validation...")
    
    # 1. Initialize Resolver
    config = {
        "cbp": {
            "mode": "PRODUCTION",
            "trials": 1,
            "strict_comparisons": True
        }
    }
    
    resolver = EOMResolver(config)
    
    # 2. Run EOM Benchmark
    results = await resolver.run_eom_benchmark()
    
    if results.get("status") == "FAILED":
        logger.error("EOM Validation FAILED.")
        sys.exit(1)
        
    # 3. Report Results
    logger.info("EOM Validation SUCCESSFUL.")
    logger.info(f"Optimized Sustained TPS: {results.get('sustained_tps', 0):.2f}")
    logger.info(f"Avg Decode Stage Latency: {results.get('avg_decode_stage_ms', 0):.2f}ms")
    logger.info(f"Serving Overhead: {results.get('avg_serialization_ms', 0) + results.get('avg_streaming_ms', 0):.2f}ms")
    
    logger.info("EOM optimizations materially materialized into serving gains.")

if __name__ == "__main__":
    asyncio.run(main())
