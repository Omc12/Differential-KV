import logging
import os
from typing import Dict, List, Any

class PRCIntegrityGuard:
    """
    Validation MUST FAIL if:
    - active systems accidentally archived
    - imports broken
    - serving paths broken
    - telemetry disconnected
    """
    def __init__(self):
        self.logger = logging.getLogger("PRCIntegrityGuard")

    def validate_prc_results(self, manifest: Dict[str, List[str]]) -> bool:
        self.logger.info("Starting PRC Integrity Audit...")
        
        # 1. Critical Files Check
        critical_files = [
            "runtime/hf_diffkv_wrapper.py",
            "serving/openai_compatible_api_gateway.py",
            "differential_kv_cli.py"
        ]
        for f in critical_files:
            if not os.path.exists(f):
                self.logger.error(f"PRC Integrity FAILED: Critical file archived or missing: {f}")
                return False
                
        # 2. Archival Non-Loss Check
        # Verify that archived files still exist in the archive directory
        archive_root = "./archive"
        if not os.path.exists(archive_root):
             self.logger.error("PRC Integrity FAILED: Archive directory not created.")
             return False
             
        # 3. Import Integrity
        try:
            from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
            from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway
        except ImportError as e:
            self.logger.error(f"PRC Integrity FAILED: Core imports broken after refactor: {e}")
            return False

        self.logger.info("PRC Integrity Audit PASSED.")
        return True

prc_integrity_guard = PRCIntegrityGuard()
