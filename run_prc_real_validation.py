import logging
import asyncio
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from runtime.prc_resolver import PRCResolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PRCValidation")

async def main():
    logger.info("Starting Phase 37.5 PRC (Platform Refactor & Consolidation) Validation...")
    
    config = {
        "prc": {
            "mode": "PRODUCTION",
            "archival_enabled": True,
            "documentation_enabled": True
        }
    }
    
    resolver = PRCResolver(config)
    
    # Run PRC Benchmark
    results = await resolver.run_prc_benchmark()
    
    if results.get("status") == "FAILED":
        logger.error("PRC Validation FAILED. Structural integrity or functional regression detected.")
        sys.exit(1)
        
    # Report Results
    logger.info("PRC Validation SUCCESSFUL.")
    logger.info(f"Files Archived: {results.get('files_archived')}")
    logger.info(f"Package Structure Valid: {results.get('package_structure_valid')}")
    logger.info(f"Telemetry Unified: {results.get('telemetry_unified')}")
    
    logger.info("Differential KV is now structurally clean and ready for Stage 2.")

if __name__ == "__main__":
    asyncio.run(main())
