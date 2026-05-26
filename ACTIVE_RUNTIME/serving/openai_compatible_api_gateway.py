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
            
            # Dynamic matching by prompt message history prefix (essential for standard OpenAI clients like Open WebUI)
            if session_id is None and self.session_manager is not None:
                incoming_messages = [{"role": m.role, "content": m.content} for m in request.messages]
                if len(incoming_messages) > 1:
                    prefix_history = incoming_messages[:-1]
                    for sid, history in getattr(self.session_manager, "message_histories", {}).items():
                        if len(history) == len(prefix_history):
                            match = True
                            for h_msg, p_msg in zip(history, prefix_history):
                                if h_msg.get("role") != p_msg.get("role") or h_msg.get("content") != p_msg.get("content"):
                                    match = False
                                    break
                            if match:
                                session_id = sid
                                print(f"[DiffKV Gateway] Dynamically matched message history prefix to active session: {session_id}")
                                break

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
                        self.session_manager.clear_history(session_id)
                        for msg in payload.get("messages", []):
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
            serving_mode = "balanced"
            model_id = "diffkv-serving"
            if hasattr(self.resolver, "wrapper") and self.resolver.wrapper:
                w = self.resolver.wrapper
                serving_mode = getattr(getattr(w, "manager", None), "serving_mode", "balanced")
                model_id = getattr(w, "model_id", model_id)
            return {
                "vram_allocated_gb": round(vram_gb, 3),
                "cuda_available":    torch.cuda.is_available(),
                "sampling_mode":     "temperature+top_p+repetition_penalty",
                "streaming_mode":    "phrase_group_chunked",
                "serving_mode":      serving_mode,
                "model":             model_id,
            }

        @self.app.get("/health")
        @self.app.get("/v1/health")
        async def health_check():
            """Health check endpoint required by Open WebUI and Ollama-compatible clients."""
            return {"status": "ok"}

        @self.app.get("/")
        async def root():
            """Root endpoint — returns service identity for discoverability."""
            return {"service": "Differential KV API", "status": "running", "docs": "/docs"}

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
                self.session_manager.clear_history(session_id)
                for msg in payload.get("messages", []):
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
    parser.add_argument('--rank', type=int, default=16,
                        help='SVD rank for KV compression. Higher = better quality, more VRAM. '
                             'Recommended: 16 for balanced, 32 for quality, 8 for VRAM-constrained.')
    parser.add_argument('--micro-block-size', type=int, default=32,
                        help='Tokens accumulated before compression is submitted. '
                             'Larger = fewer SVD calls, lower relative error. Must be >= rank.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--serving-mode', type=str,
                        choices=['lightweight', 'balanced', 'performance', 'long-context', 'fused-sparse'],
                        default='balanced',
                        help='KV cache serving mode. Use long-context for >8K tokens; fused-sparse for max GPU throughput.')
    # ── Weight quantization args ────────────────────────────────────────────────────
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load model weights in 4-bit NF4 quantization (bitsandbytes). '
                             'Reduces weight VRAM by ~70%% (e.g. Qwen2.5-1.5B: 3.1 GB -> ~0.9 GB). '
                             'Requires: pip install bitsandbytes')
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Load model weights in 8-bit LLM.int8 quantization (bitsandbytes). '
                             'Reduces weight VRAM by ~50%% with near-lossless quality. '
                             'Requires: pip install bitsandbytes')
    # ── Session residency arg ──────────────────────────────────────────────────────
    parser.add_argument('--max-resident-sessions', type=int, default=3,
                        help='Maximum number of sessions resident in VRAM simultaneously. '
                             'Lower = less VRAM used for KV cache per idle session. '
                             'Increase for multi-user serving with active parallel sessions.')
    args = parser.parse_args()

    # Disable tokenizer parallelism warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # ── Build quantization config ──────────────────────────────────────────────────
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("[DiffKV] Weight quantization: 4-bit NF4 (bitsandbytes) — weight VRAM reduced ~70%")
            elif args.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("[DiffKV] Weight quantization: 8-bit LLM.int8 (bitsandbytes) — weight VRAM reduced ~50%")
        except ImportError:
            print("[DiffKV] WARNING: bitsandbytes not installed. Falling back to full precision.")
            print("[DiffKV]   Install with: pip install bitsandbytes")
            quantization_config = None

    print(f'Loading DiffKV runtime with model: {args.model}...')
    print(f'  rank={args.rank}  micro_block_size={args.micro_block_size}  serving_mode={args.serving_mode}')
    print(f'  max_resident_sessions={args.max_resident_sessions}  quantization={"4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "none")}')
    wrapper = DiffKVHFWrapper(
        args.model,
        config={
            'rank': args.rank,
            'micro_block_size': args.micro_block_size,
            'serving_mode': args.serving_mode,
        },
        device='cuda',
        quantization_config=quantization_config,
    )
    
    print('Starting Continuous Batching Engine...')
    engine = ContinuousBatchEngine(wrapper, max_batch_size=args.batch_size)
    
    print('Starting Session Manager...')
    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=args.max_resident_sessions,
    )
    
    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    
    uvicorn.run(gateway.app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
