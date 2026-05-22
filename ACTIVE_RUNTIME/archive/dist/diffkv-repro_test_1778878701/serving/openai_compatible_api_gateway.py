import time
import uuid
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from serving.production_session_manager import ProductionSessionManager
from serving.sparse_request_scheduler import SparseRequestScheduler
from serving.serving_fault_recovery_engine import ServingFaultRecoveryEngine

# API Schemas (simplified for this implementation)
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
    """
    OpenAI-compatible API Gateway for Differential KV.
    Integrates session management, scheduling, and fault recovery.
    """
    def __init__(self, runtime_executor_fn):
        self.app = FastAPI(title="Differential KV PSI Gateway")
        self.session_manager = ProductionSessionManager()
        self.scheduler = SparseRequestScheduler()
        self.recovery_engine = ServingFaultRecoveryEngine()
        self.runtime_executor_fn = runtime_executor_fn
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            session_id = request.session_id or self.session_manager.create_session()
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            
            # Prepare prompt
            prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
            prompt += "\nassistant: "
            
            payload = {
                "prompt": prompt,
                "max_tokens": request.max_tokens or 100,
                "temperature": request.temperature
            }

            if request.stream:
                return StreamingResponse(
                    self._stream_generator(session_id, request_id, created_time, request.model, payload),
                    media_type="text/event-stream"
                )
            else:
                # Route through recovery engine and scheduler
                result = await self.recovery_engine.execute_with_recovery(
                    session_id,
                    self.scheduler.submit_request,
                    session_id,
                    payload
                )
                
                return self._format_response(request_id, created_time, request.model, result)

        @self.app.get("/v1/sessions")
        async def list_sessions():
            return {"sessions": self.session_manager.list_sessions()}

        @self.app.get("/v1/metrics")
        async def get_metrics():
            return {
                "scheduler": self.scheduler.get_serving_metrics(),
                "recovery": self.recovery_engine.get_recovery_metrics()
            }

    async def _stream_generator(self, session_id, request_id, created, model, payload):
        # In a real implementation, the scheduler/runtime would yield chunks.
        # Here we simulate the streaming flow through the infrastructure.
        
        result = await self.recovery_engine.execute_with_recovery(
            session_id,
            self.scheduler.submit_request,
            session_id,
            payload
        )
        
        text = result.get("text", "")
        chunks = text.split()
        
        for i, chunk in enumerate(chunks):
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk + " "},
                    "finish_reason": None if i < len(chunks) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.02)
        
        yield "data: [DONE]\n\n"

    def _format_response(self, request_id, created, model, result):
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("text", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            }
        }

    async def start(self):
        await self.scheduler.start(self.runtime_executor_fn)

    async def stop(self):
        await self.scheduler.stop()

if __name__ == "__main__":
    import uvicorn
    # Mock runtime for standalone testing
    async def mock_runtime(session_ids, payloads):
        results = []
        for p in payloads:
            results.append({
                "text": "This is a mock response from the PSI infrastructure.",
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            })
        return results

    gateway = OpenAICompatibleAPIGateway(mock_runtime)
    asyncio.run(gateway.start())
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
