import time
import uuid
import json
import asyncio
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.15
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class OpenAICompatibleAPIGateway:
    """
    Thin API gateway.
    Path: client -> gateway -> LGSResolver (sampling) -> streamer -> client
    No stub engines, no telemetry wrappers.
    """

    def __init__(self, resolver, session_manager=None):
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if hasattr(resolver, 'start'):
                resolver.start()
                print("Continuous Batching Engine started.")
            yield
            if hasattr(resolver, 'stop'):
                await resolver.stop()
                print("Continuous Batching Engine stopped.")

        self.app = FastAPI(title="Differential KV API", lifespan=lifespan)
        self.resolver = resolver
        self.session_manager = session_manager
        self._setup_routes()

    def _setup_routes(self):

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            # Create or reuse a session
            session_id = request.session_id
            if session_id is None and self.session_manager is not None:
                session_id = self.session_manager.create_session()
            elif session_id is None:
                session_id = str(uuid.uuid4())

            request_id   = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())

            payload = {
                "messages":           [{"role": m.role, "content": m.content} for m in request.messages],
                "max_tokens":         request.max_tokens if request.max_tokens is not None else 512,
                "temperature":        request.temperature if request.temperature is not None else 0.7,
                "top_p":              request.top_p if request.top_p is not None else 0.9,
                "repetition_penalty": request.repetition_penalty if request.repetition_penalty is not None else 1.15,
            }

            if request.stream:
                return StreamingResponse(
                    self._stream_response(
                        session_id, request_id, created_time, request.model, payload
                    ),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming chat completion
                messages = list(payload.get("messages", []))
                if self.session_manager:
                    history = self.session_manager.get_history(session_id)
                    if history and len(messages) == 1 and messages[0]["role"] == "user":
                        messages = history + messages
                        
                prompt = self.resolver.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                payload_copy = dict(payload)
                payload_copy["prompt"] = prompt
                
                try:
                    queue = await self.resolver.submit(session_id, payload_copy)
                    full_text = []
                    while True:
                        chunk = await queue.get()
                        if "error" in chunk:
                            return {"error": chunk["error"]}
                        if chunk.get("text"):
                            full_text.append(chunk["text"])
                        if chunk.get("is_final"):
                            break
                            
                    result_text = "".join(full_text)
                    
                    # Store in session manager
                    if self.session_manager:
                        original_messages = payload.get("messages", [])
                        history = self.session_manager.get_history(session_id)
                        if not history:
                            for msg in original_messages:
                                self.session_manager.append_message(session_id, msg["role"], msg["content"])
                        self.session_manager.append_message(session_id, "assistant", result_text)
                        
                    result = {"text": result_text}
                    return self._format_non_stream(request_id, created_time, request.model, result)
                except asyncio.CancelledError:
                    if hasattr(self.resolver, "cancel"):
                        self.resolver.cancel(session_id)
                    raise

        @self.app.post("/v1/sessions")
        async def create_session():
            if self.session_manager:
                sid = self.session_manager.create_session()
            else:
                sid = str(uuid.uuid4())
            return {"session_id": sid}

        @self.app.delete("/v1/sessions/{session_id}")
        async def delete_session(session_id: str):
            if self.session_manager:
                self.session_manager.clear_history(session_id)
            return {"status": "cleared", "session_id": session_id}

        @self.app.get("/v1/models")
        @self.app.get("/models")
        async def list_models():
            model_id = "diffkv-serving"
            if hasattr(self.resolver, "wrapper") and self.resolver.wrapper:
                model_id = getattr(self.resolver.wrapper, "model_id", model_id)
            elif hasattr(self.resolver, "resolver") and self.resolver.resolver:
                w = getattr(self.resolver.resolver, "wrapper", None)
                if w:
                    model_id = getattr(w, "model_id", model_id)
            
            # Sanitize model ID for Open WebUI (remove slashes which can confuse its registry parser)
            model_id = model_id.replace("/", "-")
            
            # Ensure model ID starts with diffkv-
            if not model_id.startswith("diffkv-"):
                model_id = f"diffkv-{model_id}"
                
            return {
                "object": "list",
                "data": [{
                    "id": model_id, 
                    "name": model_id, 
                    "object": "model",
                    "created": int(time.time()), 
                    "owned_by": "differential-kv"
                }],
            }

        @self.app.get("/v1/runtime_info")
        async def runtime_info():
            vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            return {
                "vram_allocated_gb": round(vram_gb, 3),
                "cuda_available":    torch.cuda.is_available(),
                "sampling_mode":     "temperature+top_p+repetition_penalty",
                "streaming_mode":    "phrase_group_chunked",
            }

    # -----------------------------------------------------------------------
    # Streaming helper
    # -----------------------------------------------------------------------

    async def _stream_response(
        self,
        session_id: str,
        request_id: str,
        created: int,
        model: str,
        payload: Dict,
    ) -> AsyncGenerator[bytes, None]:
        
        # Build full prompt with session history
        messages = list(payload.get("messages", []))
        if self.session_manager:
            history = self.session_manager.get_history(session_id)
            if history and len(messages) == 1 and messages[0]["role"] == "user":
                messages = history + messages
                
        prompt = self.resolver.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Override payload prompt for the engine
        payload_copy = dict(payload)
        payload_copy["prompt"] = prompt
        
        try:
            # Submit to background continuous batching engine
            queue = await self.resolver.submit(session_id, payload_copy)
            
            full_text = []
            while True:
                chunk = await queue.get()
                
                if "error" in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n".encode()
                    break
                    
                text = chunk.get("text", "")
                if text:
                    full_text.append(text)
                    
                finish_reason = "stop" if chunk.get("is_final") else None
                data = {
                    "id":      request_id,
                    "object":  "chat.completion.chunk",
                    "created": created,
                    "model":   model,
                    "choices": [{
                        "index":         0,
                        "delta":         {"content": text},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n".encode()
                
                if chunk.get("is_final"):
                    break

            yield b"data: [DONE]\n\n"
            
            # Store in session manager
            if self.session_manager:
                original_messages = payload.get("messages", [])
                history = self.session_manager.get_history(session_id)
                if not history:
                    for msg in original_messages:
                        self.session_manager.append_message(session_id, msg["role"], msg["content"])
                self.session_manager.append_message(session_id, "assistant", "".join(full_text))
        except asyncio.CancelledError:
            if hasattr(self.resolver, "cancel"):
                self.resolver.cancel(session_id)
            raise

    # -----------------------------------------------------------------------

    # Non-streaming response formatter
    # -----------------------------------------------------------------------

    def _format_non_stream(self, request_id, created, model, result):
        return {
            "id":      request_id,
            "object":  "chat.completion",
            "created": created,
            "model":   model,
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": result.get("text", "")},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens":      result.get("total_tokens", 0),
            },
        }

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    import argparse
    import uvicorn
    import os
    import sys
    
    _runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _runtime_dir not in sys.path:
        sys.path.insert(0, _runtime_dir)
        
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    # Disable tokenizer parallelism warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print(f'Loading DiffKV runtime with model: {args.model}...')
    wrapper = DiffKVHFWrapper(args.model, config={'rank': args.rank}, device='cuda')
    
    print('Starting Continuous Batching Engine...')
    engine = ContinuousBatchEngine(wrapper, max_batch_size=args.batch_size)
    
    print('Starting Session Manager...')
    session_manager = ProductionSessionManager()
    
    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    
    uvicorn.run(gateway.app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
