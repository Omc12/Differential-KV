import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import psutil
# Environment defaults for MPS/Metal are managed dynamically by the config preset.


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
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.15
    session_id: Optional[str] = None
    srl_enabled: Optional[bool] = None
    srl_threshold: Optional[int] = None
    srl_k_min: Optional[int] = None
    srl_k_max: Optional[int] = None
    srl_age_penalty: Optional[float] = None


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

            # Start background session cleanup task to prevent VRAM / block pool depletion
            cleanup_task = None
            if session_manager is not None:
                import asyncio
                async def _cleanup_loop():
                    while True:
                        try:
                            # Prune idle sessions every 60 seconds.
                            # Timeout is 600 seconds (10 minutes).
                            await asyncio.sleep(60)
                            session_manager.cleanup_idle_sessions(idle_timeout_seconds=600)
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            print(f"[DiffKV Cleanup] Error during idle cleanup: {e}")
                cleanup_task = asyncio.create_task(_cleanup_loop())

            # ── torch.compile warmup pass ────────────────────────────────────
            # Pre-trigger JIT compilation for the standard 512-token chunk size
            # so the first real user request doesn't absorb 30-60s of compile time.
            # Runs asynchronously so the server is immediately ready to accept
            # connections (warmup completes in the background).
            wrapper = getattr(resolver, 'wrapper', None)
            if wrapper is not None and hasattr(wrapper, 'model'):
                import asyncio, os, torch
                async def _warmup():
                    try:
                        use_compile = os.environ.get("DIFFKV_USE_TORCH_COMPILE", "auto")
                        if use_compile == "0":
                            return
                        model = wrapper.model
                        # Check if any MLP layer is compiled
                        is_compiled = False
                        layers = getattr(model, "model", model).layers if hasattr(getattr(model, "model", model), "layers") else []
                        for layer in layers:
                            if hasattr(layer, "mlp") and (hasattr(layer.mlp, "_orig_mod") or hasattr(layer.mlp, "_dynamo_ctx")):
                                is_compiled = True
                                break
                        if not hasattr(model, '_orig_mod') and not hasattr(model, '_dynamo_ctx') and not is_compiled:
                            return
                        print("[DiffKV] Running torch.compile warmup (chunk_size=512)...")
                        device = wrapper.device
                        chunk_size = 512
                        dummy_ids = torch.zeros((1, chunk_size), dtype=torch.long, device=device)
                        dummy_pos = torch.arange(0, chunk_size, dtype=torch.long, device=device).unsqueeze(0)
                        # Inject a dummy session so the attention patch doesn't error
                        wrapper.model._diffkv_session_ids = ["__warmup__"]
                        with torch.no_grad():
                            wrapper.model(input_ids=dummy_ids, position_ids=dummy_pos, use_cache=True)
                        # Clean up warmup session artifacts
                        if hasattr(wrapper, 'manager') and hasattr(wrapper.manager, 'clear_session'):
                            wrapper.manager.clear_session("__warmup__")
                        print("[DiffKV] torch.compile warmup complete.")
                    except Exception as e:
                        print(f"[DiffKV] WARNING: torch.compile warmup failed ({e}). Continuing in eager mode.")
                asyncio.create_task(_warmup())

            yield
            if cleanup_task is not None:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass

            if hasattr(resolver, 'stop'):
                await resolver.stop()
                print("Continuous Batching Engine stopped.")

        self.app = FastAPI(title="Differential KV API", lifespan=lifespan)
        self.resolver = resolver
        self.session_manager = session_manager
        self._setup_routes()

    def _is_ephemeral_request(self, messages: list) -> bool:
        """
        Detect ephemeral (title/summarization) requests that should NOT be matched
        to an active conversation session. These requests must be routed to isolated
        temporary sessions to prevent corruption of the main session's KV cache and
        prefix registry.

        Open WebUI (and similar clients) send background title/summarization requests
        immediately after each assistant turn. These follow a pattern:
          - Single user message (no prior assistant turns in THIS request)
          - Message asks for title/summary/label generation in <500 chars
          - OR the message contains Open WebUI's standard title generation prompt
        """
        if not messages:
            return False
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return False
        content = last_msg.get("content", "")
        content_lower = content.lower()

        # Check for title/summarization/naming instructions
        EPHEMERAL_KEYWORDS = (
            "title", "summarize", "summary", "label", "name this", "name the", "name of the"
        )
        has_keyword = any(kw in content_lower for kw in EPHEMERAL_KEYWORDS)
        
        # Check if the prompt contains embedded history (has assistant role markers)
        has_assistant_marker = (
            "assistant:" in content_lower or
            "assistant\n" in content_lower or
            "<|im_start|>assistant" in content_lower or
            "assistant role" in content_lower or
            "bot:" in content_lower or
            "ai:" in content_lower or
            "response:" in content_lower
        )
        
        # Heuristic 1: single message containing instructions and assistant responses (embedded history)
        if len(messages) == 1 and has_keyword and has_assistant_marker:
            return True
            
        # Heuristic 2: short explicit title request (<500 chars)
        if len(content) < 500 and has_keyword:
            return True

        # Fallback to standard long keywords matching
        EPHEMERAL_KEYWORDS_LONG = (
            "summarizing the chat history",
            "concise, 3-5 word title",
            "concise title with an emoji",
            "emoji summarizing the chat",
            "generate a concise, 3-5 word title",
            "generate a concise title with an emoji",
            'json format: { "title":',
            'json format: {"title":',
            "guidelines:\n- the title should clearly represent",
        )
        if any(kw in content_lower for kw in EPHEMERAL_KEYWORDS_LONG):
            return True

        return False

    def _optimize_ephemeral_prompt(self, content: str) -> str:
        """
        Truncate the chat history in long background title generation requests
        to make them run instantly and consume negligible memory/computation,
        without sacrificing title quality.
        """
        if len(content) <= 3000:
            return content

        content_lower = content.lower()
        for marker in ("chat history:", "conversation history:", "history:", "messages:", "context:"):
            idx = content_lower.find(marker)
            if idx != -1:
                prefix = content[:idx + len(marker)]
                history = content[idx + len(marker):]
                if len(history) > 2000:
                    history = history[:1000] + "\n... [TRUNCATED FOR SPEED] ...\n" + history[-1000:]
                print(f"[DiffKV Gateway] Optimized long title/summary prompt by truncating embedded chat history "
                      f"(shrank {len(content)} -> {len(prefix) + len(history)} characters).")
                return prefix + history

        # Fallback: simple truncation of the middle
        truncated = content[:1500] + "\n... [TRUNCATED FOR SPEED] ...\n" + content[-1500:]
        print(f"[DiffKV Gateway] Optimized long title/summary prompt by middle-truncation "
              f"(shrank {len(content)} -> {len(truncated)} characters).")
        return truncated

    def _setup_routes(self):

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            # Create or reuse a session
            session_id = request.session_id
            
            # ── Ephemeral request detection ─────────────────────────────────────
            # Detect title/summarization background requests (e.g. from Open WebUI)
            # and route them to a temporary isolated session. This prevents them from
            # matching the main conversation's history and corrupting the KV prefix
            # registry with a title-generation response.
            incoming_messages = [{"role": m.role, "content": m.content} for m in request.messages]
            is_ephemeral = self._is_ephemeral_request(incoming_messages)
            if is_ephemeral:
                orig_session_id = session_id or str(uuid.uuid4())
                ephemeral_session_id = f"__ephemeral__{orig_session_id}"
                print(f"[DiffKV Gateway] Detected ephemeral title/summary request. "
                      f"Routing to isolated session {ephemeral_session_id} to protect main KV cache.")
                session_id = ephemeral_session_id
                # Optimize/truncate the long user prompt (containing chat history) for the ephemeral request
                for m in incoming_messages:
                    if m["role"] == "user":
                        m["content"] = self._optimize_ephemeral_prompt(m["content"])
            
            # Dynamic matching by prompt message history prefix (essential for standard OpenAI clients like Open WebUI)
            if not is_ephemeral and session_id is None and self.session_manager is not None:
                if len(incoming_messages) > 1:
                    prefix_history = incoming_messages[:-1]
                    for sid, history in getattr(self.session_manager, "message_histories", {}).items():
                        # Never match against ephemeral sessions
                        if sid.startswith("__ephemeral__"):
                            continue
                        if len(history) == len(prefix_history):
                            match = True
                            for h_msg, p_msg in zip(history, prefix_history):
                                # Strip trailing/leading whitespace and newlines for robust matching
                                h_content = h_msg.get("content", "").strip()
                                p_content = p_msg.get("content", "").strip()
                                if h_msg.get("role") != p_msg.get("role") or h_content != p_content:
                                    match = False
                                    break
                            if match:
                                session_id = sid
                                print(f"[DiffKV Gateway] Dynamically matched message history prefix to active session: {session_id}")
                                break
                    
                    # Fallback match comparing the last assistant message content in the history
                    if session_id is None:
                        last_incoming_assistant = None
                        for msg in reversed(prefix_history):
                            if msg.get("role") == "assistant":
                                last_incoming_assistant = msg
                                break
                        if last_incoming_assistant is not None:
                            for sid, history in getattr(self.session_manager, "message_histories", {}).items():
                                # Never match against ephemeral sessions
                                if sid.startswith("__ephemeral__"):
                                    continue
                                last_stored_assistant = None
                                for msg in reversed(history):
                                    if msg.get("role") == "assistant":
                                        last_stored_assistant = msg
                                        break
                                if last_stored_assistant is not None:
                                    # Strip trailing/leading whitespace and newlines for robust matching
                                    h_last = last_stored_assistant.get("content", "").strip()
                                    p_last = last_incoming_assistant.get("content", "").strip()
                                    if len(p_last) > 150 and h_last == p_last:
                                        session_id = sid
                                        print(f"[DiffKV Gateway] Dynamically matched session {session_id} using fallback last assistant message content match.")
                                        break

            if not is_ephemeral and session_id is None and self.session_manager is not None:
                session_id = self.session_manager.create_session()
            elif not is_ephemeral and session_id is None:
                session_id = str(uuid.uuid4())
            elif not is_ephemeral and self.session_manager is not None:
                # Ensure the matched or requested session is loaded and resident in VRAM.
                # Fall back to creating a new session if the session has expired or is invalid.
                session = self.session_manager.get_session(session_id)
                if session is None:
                    session_id = self.session_manager.create_session()

            # Apply dynamic SRL configurations to the session (skip for ephemeral sessions)
            kv_manager = getattr(getattr(self.resolver, "wrapper", None), "manager", None)
            if not is_ephemeral and kv_manager is not None:
                srl_config = {}
                if request.srl_enabled is not None:
                    srl_config["srl_enabled"] = request.srl_enabled
                if request.srl_threshold is not None:
                    srl_config["srl_threshold"] = request.srl_threshold
                if request.srl_k_min is not None:
                    srl_config["srl_k_min"] = request.srl_k_min
                if request.srl_k_max is not None:
                    srl_config["srl_k_max"] = request.srl_k_max
                if request.srl_age_penalty is not None:
                    srl_config["srl_age_penalty"] = request.srl_age_penalty
                
                if srl_config:
                    if hasattr(kv_manager, "set_session_config"):
                        kv_manager.set_session_config(session_id, srl_config)
                    # If SRL state already exists, update its runtime configs directly
                    srl_state = kv_manager.get_srl_state(session_id)
                    if srl_state is not None:
                        if "srl_k_min" in srl_config:
                            srl_state.k_min = srl_config["srl_k_min"]
                        if "srl_k_max" in srl_config:
                            srl_state.k_max = srl_config["srl_k_max"]
                        if "srl_threshold" in srl_config:
                            srl_state.routing_threshold = srl_config["srl_threshold"]
                        if "srl_age_penalty" in srl_config:
                            srl_state.srl_age_penalty = srl_config["srl_age_penalty"]

            request_id   = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())

            payload = {
                "messages":           incoming_messages,
                "max_tokens":         request.max_tokens if request.max_tokens is not None else 2048,
                "temperature":        request.temperature if request.temperature is not None else 0.7,
                "top_p":              request.top_p if request.top_p is not None else 0.9,
                "repetition_penalty": request.repetition_penalty if request.repetition_penalty is not None else 1.15,
            }

            if request.stream:
                return StreamingResponse(
                    self._stream_response(
                        session_id, request_id, created_time, request.model, payload,
                        is_ephemeral=is_ephemeral
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
                
                is_finished = False
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
                            is_finished = True
                            break
                            
                    result_text = "".join(full_text)
                    
                    # Store in session manager (skip for ephemeral sessions)
                    if self.session_manager and not is_ephemeral:
                        self.session_manager.clear_history(session_id)
                        for msg in payload.get("messages", []):
                            self.session_manager.append_message(session_id, msg["role"], msg["content"])
                        self.session_manager.append_message(session_id, "assistant", result_text)

                    # Update prefix registry for correct Turn 2+ prefix reuse (skip for ephemeral sessions)
                    if not is_ephemeral:
                        try:
                            next_turn_messages = list(payload.get("messages", []))
                            next_turn_messages.append({"role": "assistant", "content": result_text})
                            full_next_prompt = self.resolver.tokenizer.apply_chat_template(
                                next_turn_messages, tokenize=False, add_generation_prompt=False
                            )
                            if hasattr(self.resolver, "update_session_token_prefix"):
                                self.resolver.update_session_token_prefix(session_id, full_next_prompt)
                        except Exception as _prefix_e:
                            print(f"[DiffKV Gateway] WARNING: failed to update prefix registry: {_prefix_e}")
                    else:
                        # Clean up ephemeral session KV immediately to free VRAM
                        try:
                            if hasattr(self.resolver, "_free_session_kv"):
                                self.resolver._free_session_kv(session_id)
                            elif kv_manager is not None and hasattr(kv_manager, "clear_session"):
                                kv_manager.clear_session(session_id)
                        except Exception as _cleanup_e:
                            pass  # Non-fatal; ephemeral session will GC eventually
                        
                    result = {"text": result_text}
                    return self._format_non_stream(request_id, created_time, request.model, result)
                except asyncio.CancelledError:
                    if not is_finished:
                        if self.session_manager and full_text:
                            self.session_manager.clear_history(session_id)
                            for msg in payload.get("messages", []):
                                self.session_manager.append_message(session_id, msg["role"], msg["content"])
                            self.session_manager.append_message(session_id, "assistant", "".join(full_text))
                        if hasattr(self.resolver, "cancel"):
                            self.resolver.cancel(session_id, free_kv=False)
                    raise
                finally:
                    if is_ephemeral:
                        try:
                            if hasattr(self.resolver, "_free_session_kv"):
                                self.resolver._free_session_kv(session_id)
                            elif kv_manager is not None and hasattr(kv_manager, "clear_session"):
                                kv_manager.clear_session(session_id)
                        except Exception:
                            pass
                    import gc
                    from native_core.mac_utils import empty_cache
                    gc.collect()
                    empty_cache()

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
                if hasattr(self.session_manager, "delete_session"):
                    self.session_manager.delete_session(session_id)
                else:
                    self.session_manager.clear_history(session_id)
            if hasattr(self.resolver, "cancel"):
                self.resolver.cancel(session_id)
            return {"status": "deleted", "session_id": session_id}

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
            import torch as _t
            import psutil
            _cuda_avail = _t.cuda.is_available()
            _mps_avail  = (
                hasattr(_t.backends, "mps") and _t.backends.mps.is_available()
            )
            
            # Process memory info
            process = psutil.Process()
            rss_gb = process.memory_info().rss / 1e9
            vms_gb = process.memory_info().vms / 1e9
            
            # Hardware specific memory info
            mps_allocated_gb = 0.0
            mps_driver_gb = 0.0
            cuda_allocated_gb = 0.0
            cuda_reserved_gb = 0.0
            
            device = "cpu"
            kv_manager = None
            serving_mode = "balanced"
            model_id = "diffkv-serving"
            
            if hasattr(self.resolver, "wrapper") and self.resolver.wrapper:
                w = self.resolver.wrapper
                device = getattr(w, "device", "cpu")
                kv_manager = getattr(w, "manager", None)
                serving_mode = getattr(kv_manager, "serving_mode", "balanced") if kv_manager else "balanced"
                model_id = getattr(w, "model_id", model_id)
                
            if device == "mps":
                try:
                    mps_allocated_gb = _t.mps.current_allocated_memory() / 1e9
                    mps_driver_gb = _t.mps.driver_allocated_memory() / 1e9
                except Exception:
                    pass
            elif device == "cuda":
                try:
                    cuda_allocated_gb = _t.cuda.memory_allocated() / 1e9
                    cuda_reserved_gb = _t.cuda.memory_reserved() / 1e9
                except Exception:
                    pass
            
            # Manager runtime summary
            kv_summary = {}
            if kv_manager is not None:
                try:
                    kv_summary = kv_manager.runtime_summary()
                except Exception:
                    pass

            # Sanitize NaN/Inf floats before JSON serialization.
            # torchao quantization and early compression can produce NaN in
            # avg_cosine_sim which causes a 500 error and breaks the memory monitor.
            import math
            def _sanitize(obj):
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return 0.0
                    return obj
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                return obj

            kv_summary = _sanitize(kv_summary)

            return {
                "vram_allocated_gb": round(cuda_allocated_gb if device == "cuda" else mps_allocated_gb, 3),
                "cuda_available":    _cuda_avail,
                "mps_available":     _mps_avail,
                "sampling_mode":     "temperature+top_p+repetition_penalty",
                "streaming_mode":    "phrase_group_chunked",
                "serving_mode":      serving_mode,
                "model":             model_id,
                # New fields for deep memory analysis
                "device":            device,
                "process_rss_gb":    round(rss_gb, 3),
                "process_vms_gb":    round(vms_gb, 3),
                "mps_allocated_gb":  round(mps_allocated_gb, 3),
                "mps_driver_gb":     round(mps_driver_gb, 3),
                "cuda_allocated_gb": round(cuda_allocated_gb, 3),
                "cuda_reserved_gb":  round(cuda_reserved_gb, 3),
                "kv_summary":        kv_summary
            }

        @self.app.get("/v1/sessions/{session_id}/srl")
        async def session_srl_info(session_id: str):
            """Get SRL (Semantic Routing Layer) stats for a given session."""
            kv_manager = getattr(getattr(self.resolver, "wrapper", None), "manager", None)
            if kv_manager is None:
                return {"error": "KV manager not initialized"}
            
            srl_state = kv_manager.get_srl_state(session_id)
            session_config = getattr(kv_manager, "session_configs", {}).get(session_id, {})
            _device = "cuda"
            if hasattr(self.resolver, "wrapper") and self.resolver.wrapper:
                _device = getattr(self.resolver.wrapper, "device", "cuda")
            default_threshold = 50

            if srl_state is None:
                return {
                    "session_id": session_id,
                    "srl_built": False,
                    "reason": "SRL not built yet (prefill not completed, or sequence too short)",
                    "srl_enabled": session_config.get("srl_enabled", True),
                    "k_min": session_config.get("srl_k_min", 20),
                    "k_max": session_config.get("srl_k_max", 200),
                    "routing_threshold": session_config.get("srl_threshold", default_threshold),
                    "srl_age_penalty": session_config.get("srl_age_penalty", 0.01),
                }
            
            # Extract stats safely
            n_blocks = srl_state.n_active_blocks()
            return {
                "session_id": session_id,
                "srl_built": True,
                "active_blocks": n_blocks,
                "k_min": srl_state.k_min,
                "k_max": srl_state.k_max,
                "routing_threshold": srl_state.routing_threshold,
                "srl_age_penalty": srl_state.srl_age_penalty,
                "call_count": srl_state.call_count,
                "current_step_count": srl_state.current_step_count,
                "miss_rate": round(srl_state.recent_miss_rate, 4),
                "k_multiplier": round(srl_state.k_multiplier, 4),
                "sink_blocks": srl_state.sink_blocks,
                "ordered_slot_ids": srl_state.ordered_slot_ids,
                "vocab_size": len(srl_state.inverted_index.important_vocab),
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
        is_ephemeral: bool = False,
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
        
        is_finished = False
        try:
            # Submit to background continuous batching engine
            queue = await self.resolver.submit(session_id, payload_copy)
            
            full_text = []
            # ── SSE Keepalive: prevent HTTP connection timeouts during long prefill ──
            # On MPS, prefilling a long prompt (e.g. 6K tokens) can take 30-60 seconds
            # with zero output — causing uvicorn/nginx/clients to kill the connection.
            # We poll the queue with a 5-second timeout and send SSE comment pings
            # (":ping\n\n") to keep the connection alive. These are valid in SSE spec
            # and are silently ignored by OpenAI-compatible clients.
            KEEPALIVE_INTERVAL_S = 5.0
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL_S)
                except asyncio.TimeoutError:
                    # No token yet — send keepalive comment to prevent connection drop
                    yield b": ping\n\n"
                    continue
                
                if "error" in chunk:
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n".encode()
                    break
                    
                text = chunk.get("text", "")
                if text:
                    full_text.append(text)
                    
                if chunk.get("is_final"):
                    is_finished = True
                    # Store in session manager immediately before yield of final chunk (skip for ephemeral sessions)
                    if self.session_manager and not is_ephemeral:
                        self.session_manager.clear_history(session_id)
                        for msg in payload.get("messages", []):
                            self.session_manager.append_message(session_id, msg["role"], msg["content"])
                        self.session_manager.append_message(session_id, "assistant", "".join(full_text))

                    # Build the full next-turn prompt (current messages + assistant response)
                    # and update the prefix token registry so Turn 2 correctly reuses the KV cache.
                    # Skip for ephemeral sessions — they must not pollute the main session registry.
                    if not is_ephemeral:
                        try:
                            next_turn_messages = list(payload.get("messages", []))
                            next_turn_messages.append({"role": "assistant", "content": "".join(full_text)})
                            full_next_prompt = self.resolver.tokenizer.apply_chat_template(
                                next_turn_messages, tokenize=False, add_generation_prompt=False
                            )
                            if hasattr(self.resolver, "update_session_token_prefix"):
                                self.resolver.update_session_token_prefix(session_id, full_next_prompt)
                        except Exception as _prefix_e:
                            print(f"[DiffKV Gateway] WARNING: failed to update prefix registry: {_prefix_e}")
                    else:
                        # Clean up ephemeral session KV immediately to free VRAM
                        try:
                            kv_mgr = getattr(getattr(self.resolver, "wrapper", None), "manager", None)
                            if hasattr(self.resolver, "_free_session_kv"):
                                self.resolver._free_session_kv(session_id)
                            elif kv_mgr is not None and hasattr(kv_mgr, "clear_session"):
                                kv_mgr.clear_session(session_id)
                        except Exception:
                            pass  # Non-fatal; ephemeral session will GC eventually

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
        except asyncio.CancelledError:
            if not is_finished:
                if self.session_manager and full_text:
                    self.session_manager.clear_history(session_id)
                    for msg in payload.get("messages", []):
                        self.session_manager.append_message(session_id, msg["role"], msg["content"])
                    self.session_manager.append_message(session_id, "assistant", "".join(full_text))
                if hasattr(self.resolver, "cancel"):
                    self.resolver.cancel(session_id, free_kv=False)
            raise
        finally:
            if is_ephemeral:
                try:
                    kv_mgr = getattr(getattr(self.resolver, "wrapper", None), "manager", None)
                    if hasattr(self.resolver, "_free_session_kv"):
                        self.resolver._free_session_kv(session_id)
                    elif kv_mgr is not None and hasattr(kv_mgr, "clear_session"):
                        kv_mgr.clear_session(session_id)
                except Exception:
                    pass
            import gc
            from native_core.mac_utils import empty_cache
            gc.collect()
            empty_cache()

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
def check_and_preload_allocator():
    import os
    import sys
    
    # Check if already preloaded
    if os.environ.get("MIMALLOC_PRELOADED") == "1":
        return
        
    mimalloc_paths = []
    preload_env_var = None
    
    if sys.platform == "darwin":
        # Bypassed on macOS because preloading libmimalloc intercepts system allocations
        # inside Apple libraries (like Metal/MPS driver) causing a SIGBUS (exit code 138) crash on startup.
        return
    elif sys.platform.startswith("linux"):
        mimalloc_paths = [
            "/usr/lib/x86_64-linux-gnu/libmimalloc.so.2",
            "/usr/lib/libmimalloc.so.2",
            "/usr/local/lib/libmimalloc.so",
        ]
        preload_env_var = "LD_PRELOAD"
        
    found_path = None
    for path in mimalloc_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path and preload_env_var:
        print(f"[DiffKV Allocator] Found mimalloc at {found_path}. Automatically preloading for memory compaction...")
        env = os.environ.copy()
        env[preload_env_var] = found_path
        env["MIMALLOC_PRELOADED"] = "1"
        try:
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
        except Exception as e:
            print(f"[DiffKV Allocator] WARNING: Failed to re-execute with mimalloc: {e}")

def main():
    import argparse
    import uvicorn
    import os
    import sys
    
    # ── Automatically detect and preload mimalloc ──
    check_and_preload_allocator()
    
    _runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _runtime_dir not in sys.path:
        sys.path.insert(0, _runtime_dir)
        
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--rank', type=int, default=32,
                        help='SVD rank for KV compression. Higher = better quality, more VRAM. '
                             'Must be strictly less than model head_dim (e.g. capped at 32 for head_dim 64 to prevent gibberish). '
                             'Recommended: 16 for balanced, 32 for quality, 8 for VRAM-constrained.')
    parser.add_argument('--micro-block-size', type=int, default=256,
                        help='Tokens per compressed KV block. S=256 gives 5.2x compression ratio. '
                             'Lower values compress more frequently (more overhead). Must be >= rank.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--serving-mode', type=str,
                        choices=['lightweight', 'balanced', 'performance', 'long-context', 'fused-sparse'],
                        default='balanced',
                        help='KV cache serving mode. Use long-context for >8K tokens; fused-sparse for max GPU throughput.')
    parser.add_argument('--preset', type=str,
                        choices=['low', 'mid', 'high'],
                        default='mid',
                        help='Hardware optimization preset (low for 8GB Mac/swapping-heavy, mid for default, high for server/RTX)')
    # ── Weight quantization args ────────────────────────────────────────────────────
    parser.add_argument('--load-in-4bit', action='store_true',
                        help='Load model weights in 4-bit NF4 quantization (bitsandbytes). '
                             'Reduces weight VRAM by ~70%% (e.g. Qwen2.5-1.5B: 3.1 GB -> ~0.9 GB). '
                             'Requires: pip install bitsandbytes')
    parser.add_argument('--load-in-8bit', action='store_true',
                        help='Load model weights in 8-bit LLM.int8 quantization (bitsandbytes). '
                             'Reduces weight VRAM by ~50%% with near-lossless quality. '
                             'Requires: pip install bitsandbytes')
    # ── Session residency arg ──────────────────────────────────────────────────────────────────────────────────
    parser.add_argument('--max-resident-sessions', type=int, default=4,
                        help='Maximum sessions resident in VRAM simultaneously. '
                             'Default 4 supports concurrent requests (e.g. title generation) without swapping. '
                             'Increase for multi-user serving with parallel active sessions.')
    parser.add_argument('--draft-model', type=str, default=None,
                        help='Optional path/name of the draft model for speculative decoding. '
                             'If provided, speculative decoding is enabled.')
    args = parser.parse_args()

    # Disable tokenizer parallelism warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Auto-detect best device: CUDA → MPS (Apple Silicon) → CPU
    try:
        from native_core.mac_utils import get_best_device as _gbd
        _best_device = _gbd()
    except ImportError:
        _best_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'[DiffKV] Auto-selected device: {_best_device}')

    # Apply global platform defaults early in environment
    if _best_device == "mps":
        if os.environ.get("DIFFKV_MPS_APPROXIMATE_ATTN") is None:
            os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
        if os.environ.get("DIFFKV_USE_TORCH_COMPILE") is None:
            os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
        print("[DiffKV] Apple Silicon/MPS detected. Automatically wired platform settings:")
        print(f"  - DIFFKV_MPS_APPROXIMATE_ATTN = {os.environ.get('DIFFKV_MPS_APPROXIMATE_ATTN')}")
        print(f"  - DIFFKV_USE_TORCH_COMPILE     = {os.environ.get('DIFFKV_USE_TORCH_COMPILE')}")

    # ── Platform-specific auto-optimization for 'low' preset ──
    if args.preset == "low":
        print(f'[DiffKV Debug] Low preset condition matched! Applying platform auto-optimizations...')
        
        # 1. Quantization auto-enable (INT8 on MPS via torchao, INT4 on CUDA via bitsandbytes)
        if not args.load_in_4bit and not args.load_in_8bit:
            if _best_device == "cuda":
                args.load_in_4bit = True
                print("[DiffKV] Low preset + CUDA: auto-enabling 4-bit weight quantization (bitsandbytes) to save VRAM")
            elif _best_device == "mps":
                print("[DiffKV] Low preset + MPS: running in FP16 to avoid torchao NaN/stability issues on MPS")
                
        # 2. Serving mode auto-adjustment
        if args.serving_mode not in ["lightweight"]:
            original_mode = args.serving_mode
            args.serving_mode = "lightweight"
            print(f"[DiffKV] Low preset: auto-adjusting serving_mode from '{original_mode}' to 'lightweight' to prevent OOM")
            
        # 3. Rank auto-adjustment
        if args.rank == 32:  # Only auto-adjust if using default rank
            args.rank = 16
            print(f"[DiffKV] Low preset: auto-adjusting rank from 32 to 16 to prevent attention OOM on long prompts")

    # ── Build quantization config (CUDA/bitsandbytes only) ──────────────────────────
    quantization_config = None
    if (args.load_in_4bit or args.load_in_8bit) and _best_device == "cuda":
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
            print("[DiffKV] WARNING: bitsandbytes not installed. Falling back to full precision on CUDA.")
            print("[DiffKV]   Install with: pip install bitsandbytes")
            quantization_config = None

    print(f'Loading DiffKV runtime with model: {args.model}...')
    print(f'[DiffKV Debug] settings: device={_best_device}, preset={args.preset}, serving_mode={args.serving_mode}, rank={args.rank}')
    print(f'  rank={args.rank}  micro_block_size={args.micro_block_size}  serving_mode={args.serving_mode}')
    print(f'  max_resident_sessions={args.max_resident_sessions}  quantization={"4bit" if args.load_in_4bit else ("8bit" if args.load_in_8bit else "none")}')
    print(f'  [Tip] Set DIFFKV_TELEMETRY=1 to enable VRAM + block state logging')
    
    # ── Configure MPS high watermark ratio early before allocator is initialized ──
    if _best_device == "mps":
        try:
            from native_core.config import DiffKVConfig
            cfg = DiffKVConfig({"preset": args.preset})
            watermark = cfg.mps_watermark
            if watermark > 0.0:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(watermark)
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(round(watermark * 0.8, 2))
                print(f"[DiffKV] Configured PYTORCH_MPS_HIGH_WATERMARK_RATIO={watermark}, LOW_WATERMARK_RATIO={round(watermark * 0.8, 2)}")
                torch.mps.set_per_process_memory_fraction(watermark)
            else:
                os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
                os.environ.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
                print("[DiffKV] MPS watermark set to default (no override)")
        except Exception as e:
            print(f"[DiffKV] WARNING: Failed to configure MPS memory defaults: {e}")

    wrapper = DiffKVHFWrapper(
        args.model,
        config={
            'rank':             args.rank,
            'micro_block_size': args.micro_block_size,
            'block_size':       args.micro_block_size,   # keep in sync
            'serving_mode':     args.serving_mode,
            'mode':             'fp16',
            'quantization':     'int4' if args.load_in_4bit else ('int8' if args.load_in_8bit else None),
            'preset':           args.preset,
        },
        device=_best_device,
        quantization_config=quantization_config,
    )
    
    draft_wrapper = None
    if args.draft_model:
        print(f"Loading speculative draft model: {args.draft_model}...")
        draft_wrapper = DiffKVHFWrapper(
            args.draft_model,
            config={
                'rank':             args.rank,
                'micro_block_size': args.micro_block_size,
                'block_size':       args.micro_block_size,   # keep in sync
                'serving_mode':     args.serving_mode,
                'mode':             'fp16',
                'quantization':     'int4' if args.load_in_4bit else ('int8' if args.load_in_8bit else None),
                'preset':           args.preset,
            },
            device=_best_device,
            quantization_config=quantization_config,
        )
    
    print('Starting Continuous Batching Engine...')
    engine = ContinuousBatchEngine(wrapper, max_batch_size=args.batch_size, draft_wrapper=draft_wrapper)
    
    print('Starting Session Manager...')
    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=args.max_resident_sessions,
    )
    
    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    
    # Increase keep-alive timeout to 300s (5 min) to prevent connection drops during
    # long prefill phases (e.g. 6K-token paper ingestion can take 60-90 seconds on MPS).
    # The default keep_alive=5s is far too short for LLM workloads.
    uvicorn.run(gateway.app, host=args.host, port=args.port,
                timeout_keep_alive=300, h11_max_incomplete_event_size=16 * 1024 * 1024)

if __name__ == '__main__':
    main()
