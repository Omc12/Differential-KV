import torch
import time
import asyncio
from typing import Dict, Any, List
from runtime.psr_resolver import PSRResolver
from real_multi_model_residency_controller import RealMultiModelResidencyController
from heavy_concurrent_decode_engine import HeavyConcurrentDecodeEngine
from serving_residency_telemetry import ServingResidencyTelemetry

class HSMResolver:
    """
    HSM System 6: HSM Resolver.
    Hard-binds PSR systems to real concurrent transformer execution.
    """
    def __init__(self, wrapper: Any):
        self.psr = PSRResolver(wrapper)
        self.residency_controller = RealMultiModelResidencyController(wrapper.model, wrapper.device)
        self.decode_engine = HeavyConcurrentDecodeEngine(wrapper)
        self.telemetry = ServingResidencyTelemetry(wrapper.device)
        
        # Enforce residency immediately on init
        self.residency_controller.enforce_serving_residency()

    async def resolve_hsm_serving_step(self, session_ids: List[str], payloads: List[Dict[str, Any]]):
        """Executes a heavy serving step with real hardware pressure."""
        
        # 1. Lock residency for active sessions
        for sid in session_ids:
            self.residency_controller.lock_residency(sid)
            
        # 2. Execute REAL concurrent decode
        # This replaces the simplified psr.resolve_serving_step with actual model calls
        start_ts = time.perf_counter()
        results = await self.decode_engine.concurrent_decode_step(session_ids, payloads)
        end_ts = time.perf_counter()
        
        runtime_ms = (end_ts - start_ts) * 1000
        
        # 3. Sample residency under load
        self.telemetry.sample_residency(len(session_ids))
        
        # 4. Update PSR telemetry as well for consistency
        self.psr.telemetry.record_overhead(runtime_ms, 2.0) # Real runtime, simulated serving overhead
        
        # 5. Unlock residency (optional, but good for cleanup in real systems)
        for sid in session_ids:
            self.residency_controller.unlock_residency(sid)
            
        return results

    def get_hsm_status(self):
        return {
            "residency": self.residency_controller.get_hsm_residency_metrics(),
            "engine": self.decode_engine.get_engine_metrics(),
            "telemetry": self.telemetry.get_residency_report()
        }
