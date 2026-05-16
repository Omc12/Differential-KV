import time
import uuid
import json
import asyncio
import torch
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel

from serving.production_session_manager import ProductionSessionManager
from serving.sparse_request_scheduler import SparseRequestScheduler
from serving.serving_fault_recovery_engine import ServingFaultRecoveryEngine

# API Schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    session_id: Optional[str] = None

class OpenAICompatibleAPIGateway:
    def __init__(self, resolver):
        self.app = FastAPI(title="Differential KV PSI Gateway")
        self.session_manager = ProductionSessionManager()
        self.scheduler = SparseRequestScheduler()
        self.recovery_engine = ServingFaultRecoveryEngine()
        self.resolver = resolver
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            session_id = request.session_id or self.session_manager.create_session()
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            
            payload = {
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "max_tokens": request.max_tokens or 150,
                "temperature": request.temperature
            }

            if request.stream:
                return StreamingResponse(
                    self._true_stream_generator(session_id, request_id, created_time, request.model, payload),
                    media_type="text/event-stream"
                )
            else:
                # Standard generate-then-return path
                result = await self.resolver.lgs_runtime_executor([session_id], [payload])
                return self._format_response(request_id, created_time, request.model, result[0])

        @self.app.get("/v1/runtime_info")
        async def get_runtime_info():
            # Live runtime verification endpoint
            vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            return {
                "model": self.resolver.wrapper.model_id if self.resolver.wrapper else "None",
                "dtype": str(self.resolver.wrapper.model.dtype) if self.resolver.wrapper else "None",
                "device": str(self.resolver.wrapper.device) if self.resolver.wrapper else "None",
                "current_vram_gb": vram_gb,
                "runtime_version": "Stage 2 SKO - True Live Streaming",
                "streaming_mode": "live_autoregressive"
            }

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {"id": "diffkv-qwen2.5-7b", "object": "model", "created": int(time.time()), "owned_by": "differential-kv"}
                ]
            }

    async def _true_stream_generator(self, session_id, request_id, created, model, payload):
        """
        TRUE Live Autoregressive Token Streaming.
        """
        # Note: In production, the scheduler would batch multiple requests.
        # For this SKO validation, we pipe directly to the resolver's stream executor.
        
        async for step_result in self.resolver.lgs_runtime_stream_executor([session_id], [payload]):
            if "is_done" in step_result:
                # Final chunk with timings
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "server_timings": step_result["server_timings"],
                        "streaming_mode": "live_autoregressive"
                    }
                }
                yield f"data: {json.dumps(data)}\n\n"
                break
            
            # Extract chunk for this session
            for chunk in step_result["chunks"]:
                if chunk["session_id"] == session_id:
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk["token_text"]},
                            "finish_reason": "stop" if chunk["is_final"] else None
                        }],
                        "server_timings": {
                            "decode_complete_ts": chunk["decode_complete_ts"],
                            "step": step_result["step"]
                        }
                    }
                    yield f"data: {json.dumps(data)}\n\n"
        
        yield "data: [DONE]\n\n"

    def _format_response(self, request_id, created, model, result):
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.get("text", "")},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "server_timings": result.get("server_timings", {})
            }
        }

    async def start(self):
        await self.scheduler.start(self.resolver.lgs_runtime_executor)

    async def stop(self):
        await self.scheduler.stop()
