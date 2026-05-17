import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
from .lgs_resolver import LGSResolver

class UnifiedRuntimePackagingLayer:
    """
    OIS Phase 40.1: Unified Runtime Packaging Layer.
    Provides structured boot, config loading, and environment verification.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.config = {}
        self.logger = self._setup_logger()
        self.resolver = None

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("UnifiedRuntime")

    def boot(self, mode: str = "production") -> bool:
        """
        Starts the runtime lifecycle.
        """
        self.logger.info(f"Booting Differential KV Runtime [Mode: {mode}]")
        
        # 1. Environment Verification
        if not self._verify_environment():
            self.logger.error("Environment verification failed.")
            return False

        # 2. Dependency Validation
        if not self._validate_dependencies():
            self.logger.error("Dependency validation failed.")
            return False

        # 3. Config Loading
        self.config = self._load_config(mode)
        
        # 4. Initialize LGS Resolver
        self.resolver = LGSResolver(self.config)
        self.resolver.setup_runtime()
        
        self.logger.info("Runtime boot successful.")
        return True

    def _verify_environment(self) -> bool:
        """Checks for GPU availability and workspace structure."""
        # Simplified for now
        required_dirs = [
            "runtime",
            "reports/stage3a/phase_40_1_ois",
            "telemetry/stage3a/phase_40_1_ois",
            "traces/stage3a/phase_40_1_ois"
        ]
        for d in required_dirs:
            p = self.workspace_root / d
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created missing directory: {d}")
        return True

    def _validate_dependencies(self) -> bool:
        """Ensures torch and triton are available if needed."""
        try:
            import torch
            self.logger.info(f"Torch version: {torch.__version__}")
            if torch.cuda.is_available():
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.warning("No GPU detected. Performance will be degraded.")
        except ImportError:
            self.logger.error("Torch not found.")
            return False
        return True

    def _load_config(self, mode: str) -> Dict[str, Any]:
        """Loads configuration based on mode."""
        # In a real system, this would load from a YAML or JSON file
        config = {
            "mode": mode,
            "max_batch_size": 16 if mode == "production" else 4,
            "timeout_seconds": 30,
            "streaming_enabled": True,
            "telemetry_interval": 1.0,
            "recovery_enabled": True,
            "prefill_chunk_size": 512
        }
        self.logger.info(f"Config loaded: {config}")
        return config

    def get_config(self) -> Dict[str, Any]:
        return self.config

    async def lgs_runtime_executor(self, session_ids: List[str], payloads: List[Dict]) -> List[Dict]:
        return await self.resolver.lgs_runtime_executor(session_ids, payloads)

    async def lgs_runtime_stream_executor(self, session_ids: List[str], payloads: List[Dict]) -> AsyncGenerator[Dict, None]:
        async for chunk in self.resolver.lgs_runtime_stream_executor(session_ids, payloads):
            yield chunk
