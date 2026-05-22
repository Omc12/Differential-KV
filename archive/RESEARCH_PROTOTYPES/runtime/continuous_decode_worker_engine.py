import torch
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

class ContinuousDecodeWorkerEngine:
    """
    STAGE 2 CDBE: Continuous Decode Worker Engine.
    Maintains a persistent decode loop that stays hot and minimizes wakeup overhead.
    """
    def __init__(self, wrapper, fusion_engine, max_batch_size: int = 128):
        self.wrapper = wrapper
        self.fusion_engine = fusion_engine
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger("CDBEWorker")
        
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.pending_queue = deque()
        self._is_running = False
        self._loop_task: Optional[asyncio.Task] = None
        
        # Telemetry hooks
        self.step_counts = 0
        self.last_batch_size = 0
        self.is_busy = False

        # DQO Instrumentation (Optional)
        self.continuity_monitor = None
        self.throughput_tracker = None
        self.efficiency_instrumentation = None

    def set_dqo_instrumentation(self, continuity_monitor, throughput_tracker, efficiency_instrumentation):
        self.continuity_monitor = continuity_monitor
        self.throughput_tracker = throughput_tracker
        self.efficiency_instrumentation = efficiency_instrumentation

    async def start(self):
        """Starts the persistent decode loop."""
        if self._is_running:
            return
        self._is_running = True
        self._loop_task = asyncio.create_task(self._persistent_decode_loop())
        self.logger.info("Continuous Decode Worker Engine STARTED.")

    async def stop(self):
        """Stops the persistent decode loop."""
        self._is_running = False
        if self._loop_task:
            await self._loop_task
        self.logger.info("Continuous Decode Worker Engine STOPPED.")

    def add_session(self, session_id: str, input_ids: torch.Tensor, max_tokens: int, output_queue: asyncio.Queue):
        """Adds a new session to the active decode pool."""
        self.active_sessions[session_id] = {
            "input_ids": input_ids,
            "max_tokens": max_tokens,
            "tokens_generated": 0,
            "output_queue": output_queue,
            "finished": False
        }
        print(f"DEBUG: Session {session_id} added to CDBE worker.", flush=True)

    async def _persistent_decode_loop(self):
        while self._is_running:
            if not self.active_sessions:
                self.is_busy = False
                await asyncio.sleep(0.001) # Minimal sleep to prevent CPU spin but remain hot
                continue
            
            self.is_busy = True
            step_start = time.time()
            
            # 1. Prepare Batch
            session_ids = list(self.active_sessions.keys())[:self.max_batch_size]
            current_batch_size = len(session_ids)
            self.last_batch_size = current_batch_size
            
            if self.continuity_monitor:
                self.continuity_monitor.record_step_start(current_batch_size)
            
            # Prepare tensors for fused decode
            batch_last_tokens = []
            for sid in session_ids:
                batch_last_tokens.append(self.active_sessions[sid]["input_ids"][:, -1:])
            
            batch_input = torch.cat(batch_last_tokens, dim=0)
            
            # 2. Fused Execution Window
            # Use the fusion engine to perform the actual GPU work
            import sys
            # sys.stderr.write(f"DEBUG: Worker starting batch for {session_ids}\n")
            # sys.stderr.flush()
            logits = self.fusion_engine.fuse_decode_batch(session_ids, batch_input)
            next_tokens = torch.argmax(logits, dim=-1)
            sys.stderr.write(f".") # Dot per token for visual progress
            sys.stderr.flush()
            
            # 3. Post-processing & Streaming Distribution
            decode_complete_ts = time.time()
            
            for i, sid in enumerate(session_ids):
                session = self.active_sessions[sid]
                token_id = next_tokens[i:i+1]
                
                # Update session state
                session["input_ids"] = torch.cat([session["input_ids"], token_id.unsqueeze(0)], dim=-1)
                session["tokens_generated"] += 1
                
                if self.throughput_tracker:
                    self.throughput_tracker.record_tokens(sid, 1)
                
                # Check for EOS or Max Tokens
                is_eos = token_id.item() == self.wrapper.tokenizer.eos_token_id
                is_max = session["tokens_generated"] >= session["max_tokens"]
                
                # Prepare token payload
                token_text = self.wrapper.tokenizer.decode(token_id, skip_special_tokens=False)
                
                # Async send to output payload (including logits for semantic validation)
                payload = {
                    "session_id": sid,
                    "token_text": token_text,
                    "decode_complete_ts": decode_complete_ts,
                    "is_final": is_eos or is_max,
                    "logits": logits[i:i+1].detach().clone(), # Cloned for safety
                    "input_ids": session["input_ids"].detach().clone()
                }
                
                # We use put_nowait because we assume the consumer is fast enough or has a large buffer
                # In a real system, we'd handle backpressure, but for CDBE we want maximum push.
                session["output_queue"].put_nowait(payload)
                
                if is_eos or is_max:
                    del self.active_sessions[sid]
            
            self.step_counts += 1
            step_end = time.time()
            
            if self.continuity_monitor:
                self.continuity_monitor.record_step_finish()
            
            if self.efficiency_instrumentation:
                # We need queue depth here. For now we use the engine's view if available, 
                # but we'll assume it's passed or tracked elsewhere.
                # Since the worker doesn't see the aggregator's queue directly, we might need to adjust.
                self.efficiency_instrumentation.record_batch_step(
                    current_batch_size, 
                    len(self.pending_queue), # This might be internal queue
                    len(self.active_sessions),
                    step_end - step_start
                )
            
            # Minimal yielding to allow asyncio to handle other tasks (e.g., networking)
            # but we want to stay on the GPU as much as possible.
            await asyncio.sleep(0)

    def get_occupancy_stats(self) -> Dict[str, Any]:
        """Returns real-time occupancy metrics."""
        return {
            "active_sessions": len(self.active_sessions),
            "total_steps": self.step_counts,
            "last_batch_size": self.last_batch_size,
            "is_busy": self.is_busy
        }
