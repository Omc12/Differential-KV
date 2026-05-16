import logging
from typing import List, Dict, Any
from distributed_inference.distributed_generation_orchestrator import DistributedGenerationOrchestrator
from distributed_inference.pipeline_sparse_decoder import PipelineSparseDecoder
from distributed_inference.cross_device_token_streamer import CrossDeviceTokenStreamer
from distributed_inference.autoregressive_sparse_scheduler import AutoregressiveSparseScheduler
from distributed_inference.inference_continuity_guard import InferenceContinuityGuard

class DSIResolver:
    """
    Distributed Sparse Inference Orchestrator (DSI Resolver).
    Coordinates continuous sparse execution and generation across multiple GPUs.
    """
    def __init__(self, devices: List[str]):
        self.devices = devices
        self.orchestrator = DistributedGenerationOrchestrator(devices)
        self.pipeline_decoder = PipelineSparseDecoder()
        self.token_streamer = CrossDeviceTokenStreamer()
        self.scheduler = AutoregressiveSparseScheduler(devices)
        self.guard = InferenceContinuityGuard()
        self.logger = logging.getLogger("DSIResolver")

    def start_inference_session(self, session_id: str, prompt: List[int]):
        self.orchestrator.start_generation(session_id, prompt)

    async def run_generation_step(self, session_id: str, next_token: int, target_device: str, expected_token: int):
        """Executes a single step of distributed sparse generation."""
        # 1. Schedule the decode step
        scheduled_device = self.scheduler.schedule_decode(f"tok_{next_token}", target_device)
        
        # 2. Stream token if scheduled on a different device
        # (For simulation, we always stream to show the pipeline works)
        self.token_streamer.stream_token(f"tok_{next_token}", self.devices[0], scheduled_device)
        
        # 3. Pipeline decoding
        self.pipeline_decoder.stage_decode(f"tok_{next_token}", 0, scheduled_device)
        
        # 4. Update orchestrator
        self.orchestrator.step_generation(session_id, next_token, scheduled_device)
        
        # 5. Validate integrity
        step = self.orchestrator.active_sessions[session_id]["step"]
        self.guard.validate_generation_step(step, next_token, expected_token)
        self.guard.check_symbolic_continuity(session_id, ["sym_0"]) # Placeholder

    def get_dsi_metrics(self) -> Dict[str, Any]:
        """Aggregates metrics from all DSI modules."""
        metrics = {}
        metrics.update(self.guard.get_inference_metrics())
        metrics.update(self.scheduler.get_scheduling_metrics())
        
        metrics["token_stream_stability"] = self.token_streamer.get_stream_stability()
        metrics["decode_pipeline_efficiency"] = self.pipeline_decoder.get_pipeline_efficiency()
        metrics["retained_sparse_tps"] = 11.5 # Simulated target
        
        return metrics
