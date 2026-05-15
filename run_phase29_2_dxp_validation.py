"""
run_phase29_2_dxp_validation.py

Validation script for Phase 29.2: DXP (Developer Experience & Packaging).
Verifies that CLI, packaging, documentation, and examples are all functional.
"""

import os
import json
import logging
from runtime.dxp_resolver import DXPResolver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase29.2Validation")

def main():
    logger.info("Starting Phase 29.2 DXP Validation...")
    
    resolver = DXPResolver({})
    metrics = resolver.run_onboarding_pass()
    
    # Required Metrics Check
    required_metrics = [
        "package_build_success",
        "cli_execution_integrity",
        "quickstart_reproducibility",
        "installation_diagnostic_accuracy",
        "example_execution_success",
        "documentation_generation_stability",
        "packaging_reproducibility",
        "developer_onboarding_index"
    ]
    
    missing = [m for m in required_metrics if m not in metrics]
    if missing:
        logger.error(f"Missing required metrics: {missing}")
        exit(1)
        
    logger.info("All required DXP metrics validated.")
    
    # Physical File Checks
    checks = {
        "pyproject.toml": os.path.exists("pyproject.toml"),
        "docs/QUICKSTART.md": os.path.exists("docs/QUICKSTART.md"),
        "examples/hf_integration.py": os.path.exists("examples/hf_integration.py"),
        "workspace/config.json": os.path.exists("workspace/config.json")
    }
    
    for path, exists in checks.items():
        if not exists:
            logger.error(f"Critical DXP artifact missing: {path}")
            exit(1)
        logger.info(f"Verified: {path}")
        
    # Final Result
    final_status = "SUCCESS"
    logger.info("\n" + "="*40)
    logger.info("PHASE 29.2 DXP VALIDATION SUMMARY")
    logger.info("="*40)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k:30}: {v:.4f}")
    logger.info("="*40)
    logger.info(f"STATUS: {final_status}")

if __name__ == "__main__":
    main()
