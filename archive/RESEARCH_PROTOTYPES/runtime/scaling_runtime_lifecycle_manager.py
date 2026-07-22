import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from runtime.sat_run_manager import SATRunManager
from runtime.hf_dkv_wrapper import DKVHFWrapper
from decode_pipeline_fusion_engine import DecodePipelineFusionEngine
from runtime.cdbe_resolver import CDBEResolver

class ScalingRuntimeLifecycleManager:
    """
    SGC Phase 39.1: Scaling Runtime Lifecycle Manager.
    Ensures clean startup, sequential model loading, and trace flushing
    for multi-model scaling validations.
    """
    def __init__(self, phase="phase_39_1_sgc", stage="stage2"):
        self.phase = phase
        self.stage = stage
        self.logger = logging.getLogger("SGC_Lifecycle")
        self.current_resolver: Optional[CDBEResolver] = None
        self.current_wrapper: Optional[DKVHFWrapper] = None
        self.current_run_mgr: Optional[SATRunManager] = None

    def create_run_manager(self, model_id: str, concurrency: int, duration_sec: int) -> SATRunManager:
        """Creates an isolated run manager for a specific model."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_slug = model_id.split("/")[-1].lower().replace(".", "_")
        run_id = f"{timestamp}_{model_slug}"
        
        run_mgr = SATRunManager(
            model_id=model_id,
            concurrency=concurrency,
            duration_sec=duration_sec,
            phase=self.phase,
            stage=self.stage,
            run_id=run_id
        )
        self.current_run_mgr = run_mgr
        return run_mgr

    async def startup_model(self, model_id: str, bnb_config: Any, block_size: int, rank: int, dtype: Any = None):
        """Loads a model and initializes the CDBE resolver."""
        import torch
        torch_dtype = dtype or torch.float16
        self.logger.info(f"--- SGC Scaling: Loading {model_id} (dtype={torch_dtype}) ---")
        
        # Ensure previous state is cleared
        if self.current_resolver:
            await self.shutdown()

        try:
            self.current_wrapper = DKVHFWrapper(
                model_id,
                {"mode": "lowrank_sparse", "block_size": block_size, "rank": rank},
                quantization_config=bnb_config,
                torch_dtype=torch_dtype
            )
            fusion_engine = DecodePipelineFusionEngine(self.current_wrapper)
            
            # Isolate telemetry for scaling
            telemetry_path = self.current_run_mgr.telemetry_path("decode_overlap.jsonl")
            self.current_resolver = CDBEResolver(self.current_wrapper, fusion_engine, telemetry_path=telemetry_path)
            
            await self.current_resolver.start()
            self.logger.info(f"Runtime Online: {model_id}")
            return self.current_resolver
        except Exception as e:
            self.logger.error(f"Failed to start model {model_id}: {e}")
            raise

    async def shutdown(self):
        """Gracefully stops the runtime and flushes all traces."""
        if self.current_resolver:
            self.logger.info("Stopping resolver and flushing traces...")
            await self.current_resolver.stop()
            self.current_resolver = None
        
        if self.current_wrapper:
            # Explicitly clear CUDA cache to make room for next model
            import torch
            del self.current_wrapper
            torch.cuda.empty_cache()
            self.current_wrapper = None
            
        self.logger.info("SGC Lifecycle: Shutdown complete.")

    def seal_run(self, summary: Dict[str, Any]):
        """Completes the run manifest."""
        if self.current_run_mgr:
            self.current_run_mgr.complete(summary=summary)
            self.logger.info(f"Run {self.current_run_mgr.run_id} sealed.")
