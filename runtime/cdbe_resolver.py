import logging
import asyncio
import time
import torch
from typing import Dict, List, Any, AsyncGenerator

from runtime.continuous_decode_worker_engine import ContinuousDecodeWorkerEngine
from runtime.dynamic_decode_batch_aggregator import DynamicDecodeBatchAggregator
from runtime.persistent_decode_queue_scheduler import PersistentDecodeQueueScheduler
from runtime.decode_overlap_telemetry import DecodeOverlapTelemetry
from serving.chunked_token_streaming_layer import ChunkedTokenStreamingLayer
from runtime.native.persistent_cuda_graph_execution_manager import PersistentCUDAGraphExecutionManager

class CDBEResolver:
    """
    STAGE 2 CDBE: Continuous Decode & Batching Engine Resolver.
    Integrates all CDBE components into a unified serving runtime.
    """
    def __init__(self, wrapper, fusion_engine):
        self.wrapper = wrapper
        self.fusion_engine = fusion_engine
        self.logger = logging.getLogger("CDBEResolver")
        
        # Initialize Components
        self.telemetry = DecodeOverlapTelemetry("telemetry/stage2/phase_38_7_cdbe/decode_overlap.jsonl")
        self.graph_manager = PersistentCUDAGraphExecutionManager()
        
        self.worker = ContinuousDecodeWorkerEngine(self.wrapper, self.fusion_engine)
        self.aggregator = DynamicDecodeBatchAggregator(self.worker)
        self.scheduler = PersistentDecodeQueueScheduler(self.aggregator, self.telemetry)
        
        self._is_running = False

    async def start(self):
        if self._is_running:
            return
        await self.worker.start()
        await self.aggregator.start()
        await self.scheduler.start()
        self._is_running = True
        self.logger.info("CDBE Resolver Infrastructure ONLINE.")

    async def stop(self):
        await self.scheduler.stop()
        await self.aggregator.stop()
        await self.worker.stop()
        self._is_running = False

    async def execute_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes a single request using the CDBE pipeline.
        """
        session_id = payload.get("session_id", f"cdbe-{int(time.time()*1000)}")
        messages = payload.get("messages", [])
        max_tokens = payload.get("max_tokens", 128)
        
        # 1. Prepare Inputs
        prompt = self.wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.wrapper.tokenizer(prompt, return_tensors='pt').to(self.wrapper.device)
        input_ids = encoded.input_ids
        
        # 2. Prefill (Non-blocking to maintain loop continuity)
        await asyncio.to_thread(self.wrapper.forward_step, input_ids, session_id=session_id)
        
        # 3. Schedule for continuous decode
        token_queue = await self.scheduler.schedule(session_id, input_ids, max_tokens)
        
        # 4. Wrap in chunked streaming layer
        streamer = ChunkedTokenStreamingLayer(token_queue, chunk_size=4, timeout_ms=50)
        
        async for chunk in streamer.stream_generator():
            yield chunk
            
        # 5. Record Completion
        self.telemetry.record_completion(session_id)
