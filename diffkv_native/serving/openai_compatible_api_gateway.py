import os
import subprocess
import asyncio
import json
import uuid
import time
import threading
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

def _normalize_references(text: str) -> str:
    """Normalise citation-list formatting inconsistencies produced by the model."""
    lines = text.split('\n')
    
    # 1. Search for a reference header line
    header_re = re.compile(r'\b(references?|bibliography|works\s+cited|reference\s+list|sources|citations)\b', re.IGNORECASE)
    header_idx = None
    for i, line in enumerate(lines):
        if len(line) <= 100 and header_re.search(line):
            header_idx = i
    
    # 2. Find matching reference entries
    ref_entry_re = re.compile(r'^(?:[iI]n\s+)?(?:[*\-•]\s*)?\[\d+\]')
    unambiguous_re = re.compile(r'^(?:[*\-•]\s*)?\[\d+\]')
    
    matching_indices = []
    unambiguous_indices = []
    for i, line in enumerate(lines):
        if header_idx is not None and i <= header_idx:
            continue
        stripped = line.strip()
        if ref_entry_re.match(stripped):
            matching_indices.append(i)
            if unambiguous_re.match(stripped):
                unambiguous_indices.append(i)
                
    if header_idx is not None and not matching_indices:
        matching_indices = []
        unambiguous_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if ref_entry_re.match(stripped):
                matching_indices.append(i)
                if unambiguous_re.match(stripped):
                    unambiguous_indices.append(i)
        header_idx = None

    if not matching_indices:
        return text
        
    if header_idx is not None:
        ref_start_idx = header_idx + 1
    elif unambiguous_indices:
        ref_start_idx = unambiguous_indices[0]
    else:
        return text
                
    body = '\n'.join(lines[:ref_start_idx])
    ref_block = '\n'.join(lines[ref_start_idx:])
    
    pattern = re.compile(
        r'^\s*'
        r'(?:[iI]n\s+)?'
        r'(?:[*\-•]\s*)?'
        r'(\[\d+\])'
        r'(?:,\s*|\.\s*|\s+)?',
        re.MULTILINE
    )
    normalized_ref_block = pattern.sub(r'\1 ', ref_block)
    
    if body:
        return body + '\n' + normalized_ref_block
    return normalized_ref_block

app = FastAPI(title="DiffKV C++ Native API Server")

# Global lock to ensure only one request uses the C++ subprocess at a time
lock = asyncio.Lock()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 16384
    temperature: Optional[float] = 0.7

# Resolve paths relative to this file (…/diffkv_native/serving/ → …/diffkv_native/)
# so the server runs from any checkout location. Overridable via env vars.
_NATIVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _model_path(filename: str) -> str:
    return os.path.join(_NATIVE_ROOT, filename)

BINARY_PATH_DEFAULT = os.environ.get(
    "DIFFKV_BINARY_PATH", os.path.join(_NATIVE_ROOT, "build", "diffkv_native"))
MODEL_PATH_DEFAULT  = os.environ.get(
    "DIFFKV_MODEL_PATH", _model_path("qwen2.5-0.5b-instruct.gguf"))

# Sentinel bytes
_SENTINEL_RESPONSE = b"__RESPONSE__"
_SENTINEL_FINISH   = b"__FINISH__"
_SENTINEL_READY    = b"__READY__"

class SubprocessWrapper:
    def __init__(self):
        self.process = None
        # Per Open-WebUI-session: last prompt tokens sent and how many are cached in KV pool
        # Matches ACTIVE_RUNTIME's session_token_ids / get_session_sequence_length mechanism
        self.session_cached_len: dict   = {}  # session_id -> int (tokens resident in binary KV pool)
        self.session_prompt_text: dict  = {}  # session_id -> last full prompt string sent
        self.active_session_id: str     = ""  # session_id currently resident in C++ subprocess cache

    def _clear_session(self, session_id: str):
        self.session_cached_len.pop(session_id, None)
        self.session_prompt_text.pop(session_id, None)


    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self.stderr_log.append(decoded)
                if len(self.stderr_log) > 100:
                    self.stderr_log.pop(0)
        except Exception:
            pass

    def start(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass

        binary_path = os.getenv("DIFFKV_BINARY_PATH", BINARY_PATH_DEFAULT)
        model_path  = os.getenv("DIFFKV_MODEL_PATH",  MODEL_PATH_DEFAULT)

        print(f"[Server] Launching C++ subprocess: {binary_path} {model_path} -")
        
        self.verbose = os.environ.get("DIFFKV_VERBOSE") == "1"
        stderr_dest = None if self.verbose else subprocess.PIPE
        self.stderr_log = []

        self.process = subprocess.Popen(
            [binary_path, model_path, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_dest,
            text=False,
            bufsize=0,
            env=os.environ,
        )

        if not self.verbose:
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()

        # Drain stdout until __READY__
        buf = b""
        while True:
            if self.process.poll() is not None:
                print("[Server] Error: Subprocess failed to start (exited early). Stderr logs:")
                for line in self.stderr_log:
                    print(f"  {line}")
                raise RuntimeError("Subprocess exited early")

            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                print("[Server] Error: Subprocess failed to start (stdout closed). Stderr logs:")
                for line in self.stderr_log:
                    print(f"  {line}")
                raise RuntimeError("Subprocess failed to start (stdout closed)")
            buf += chunk
            if _SENTINEL_READY in buf:
                break
        print("[Server] C++ Native process started and ready.")

    def stop(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.process = None

    def _write_stdin(self, text: str):
        """Encode and write a single-line prompt to the C++ binary stdin."""
        self.process.stdin.write(text.encode("utf-8"))
        self.process.stdin.flush()

    def _is_alive(self) -> bool:
        """Check if the C++ subprocess is still running."""
        return self.process is not None and self.process.poll() is None

    def ensure_alive(self):
        """Restart the C++ process if it has died."""
        if not self._is_alive():
            print("[Server] C++ process died — restarting...", flush=True)
            self.start()

    def query_stream_into_queue(self, prompt: str, max_tokens: int,
                                out_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
                                cached_len: int = 0, session_id: str = ""):
        """
        Runs in a background thread.
        Writes prompt to C++ stdin, reads stdout in chunks.
        Puts decoded text chunks into out_queue.
        Puts None when done (signals end-of-stream).
        Puts {"error": msg} dict on fatal errors.
        Puts {"cached_len": N} when __CACHED__:<N> is received after __FINISH__.

        Protocol:
          C++ stdin:  [__CACHED__:<N>\n]  prompt-line\n
          C++ stdout: ... __RESPONSE__\n <tokens...> __FINISH__\n __CACHED__:<N>\n
        """
        # Auto-restart if process has crashed between requests
        self.ensure_alive()

        # Escape newlines and backslashes so the prompt fits on one stdin line.
        single_line = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")

        # Prepend __CACHED__:<N> prefix if the binary already has tokens in its KV pool
        # (ACTIVE_RUNTIME prefix reuse: skip re-prefilling already-compressed tokens)
        if cached_len > 0:
            stdin_payload = f"__CACHED__:{cached_len}\n{single_line}\n"
        else:
            stdin_payload = single_line + "\n"

        try:
            self._write_stdin(stdin_payload)
        except BrokenPipeError:
            print("[Server] BrokenPipeError writing prompt — restarting C++ process...", flush=True)
            self._clear_session(session_id)
            try:
                self.start()
                self._write_stdin(stdin_payload)
            except Exception as e2:
                loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"failed after restart: {e2}"})
                loop.call_soon_threadsafe(out_queue.put_nowait, None)
                return
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"stdin write error: {e}"})
            loop.call_soon_threadsafe(out_queue.put_nowait, None)
            return

        # ── Phase 1: wait for __RESPONSE__ ──────────────────────────────────
        buf = b""
        try:
            while True:
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"error": "process exited before __RESPONSE__"})
                    loop.call_soon_threadsafe(out_queue.put_nowait, None)
                    return
                buf += chunk
                if _SENTINEL_RESPONSE in buf:
                    # Discard everything up to and including the sentinel + newline
                    after = buf.split(_SENTINEL_RESPONSE, 1)[1]
                    # Skip the trailing newline of __RESPONSE__\n
                    if after.startswith(b"\n"):
                        after = after[1:]
                    buf = after
                    break
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"read error during prefill: {e}"})
            loop.call_soon_threadsafe(out_queue.put_nowait, None)
            return

        # Signal to the async side that prefill is done, generation starting
        loop.call_soon_threadsafe(out_queue.put_nowait, {"prefill_done": True, "text": ""})

        # Yield any bytes that arrived with or after __RESPONSE__
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        _SENTINEL_CACHED_PREFIX = b"__CACHED__:"

        def _extract_cached_from_remainder(remainder: bytes) -> int:
            """Extract __CACHED__:<N> value from bytes following __FINISH__, returns -1 if not found."""
            idx = remainder.find(_SENTINEL_CACHED_PREFIX)
            if idx == -1:
                return -1
            end = remainder.find(b"\n", idx + len(_SENTINEL_CACHED_PREFIX))
            if end == -1:
                val_bytes = remainder[idx + len(_SENTINEL_CACHED_PREFIX):]
            else:
                val_bytes = remainder[idx + len(_SENTINEL_CACHED_PREFIX):end]
            try:
                return int(val_bytes.strip())
            except Exception:
                return -1

        def _read_cached_from_stdout(proc_stdout) -> int:
            """Read bytes from stdout until __CACHED__:<N> is found or timeout."""
            buf = b""
            for _ in range(5):
                try:
                    chunk = os.read(proc_stdout.fileno(), 100)
                    if not chunk:
                        break
                    buf += chunk
                    idx = buf.find(_SENTINEL_CACHED_PREFIX)
                    if idx != -1:
                        if b"\n" in buf[idx + len(_SENTINEL_CACHED_PREFIX):]:
                            break
                except Exception:
                    break
            val = _extract_cached_from_remainder(buf)
            return val

        if buf:
            if _SENTINEL_FINISH in buf:
                parts = buf.split(_SENTINEL_FINISH, 1)
                final_text = decoder.decode(parts[0], final=True)
                if final_text:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final_text})
                remainder = parts[1] if len(parts) > 1 else b""
                new_cached = _extract_cached_from_remainder(remainder)
                if new_cached == -1:
                    new_cached = _read_cached_from_stdout(self.process.stdout)
                if new_cached >= 0:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"cached_len": new_cached})
                    print(f"[Gateway] Binary KV pool: {new_cached} tokens cached.", flush=True)
                loop.call_soon_threadsafe(out_queue.put_nowait, None)
                return
            
            text = decoder.decode(buf)
            if text:
                loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})

        # ── Phase 2: stream tokens until __FINISH__ ──────────────────────────
        accumulated = b""
        try:
            while True:
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    break
                accumulated += chunk
                if _SENTINEL_FINISH in accumulated:
                    parts = accumulated.split(_SENTINEL_FINISH, 1)
                    final_text = decoder.decode(parts[0], final=True)
                    if final_text:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final_text})
                    # Extract __CACHED__:<N> — may be in accumulated remainder or need fresh read
                    remainder = parts[1] if len(parts) > 1 else b""
                    new_cached = _extract_cached_from_remainder(remainder)
                    if new_cached == -1:
                        # Not yet buffered — read fresh bytes from stdout
                        new_cached = _read_cached_from_stdout(self.process.stdout)
                    if new_cached >= 0:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"cached_len": new_cached})
                        print(f"[Gateway] Binary KV pool: {new_cached} tokens cached.", flush=True)
                    break
                tail_len = len(_SENTINEL_FINISH) + 4
                if len(accumulated) > tail_len:
                    safe = accumulated[:-tail_len]
                    remaining = accumulated[-tail_len:]
                    text = decoder.decode(safe)
                    if text:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})
                    accumulated = remaining
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"read error during generation: {e}"})

        loop.call_soon_threadsafe(out_queue.put_nowait, None)


def format_messages_as_chat(messages: list) -> str:
    """Build a full Qwen2.5 chat-template string from ChatMessage objects."""
    result = ""
    has_system = any(msg.role == "system" for msg in messages)
    if not has_system:
        result += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for msg in messages:
        if msg.role in ("system", "user", "assistant"):
            result += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


# Initialize dual wrappers for main conversation and ephemeral background requests
main_wrapper = SubprocessWrapper()
ephemeral_wrapper = SubprocessWrapper()

main_lock = asyncio.Lock()
ephemeral_lock = asyncio.Lock()

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    main_wrapper.start()
    # ephemeral_wrapper is started lazily upon first ephemeral request
    yield
    main_wrapper.stop()
    ephemeral_wrapper.stop()

app.router.lifespan_context = lifespan


# ── Ephemeral request detection (matching ACTIVE_RUNTIME) ────────────────────
# Open WebUI sends automatic title/summary requests after each turn.
# These embed the full conversation history (can be 30KB+) and would cause
# a full prefill of 8000+ tokens — hanging or taking minutes.
# We detect and truncate them before they hit the C++ binary.

_EPHEMERAL_KEYWORDS = (
    "title", "label", "name this", "name the",
    "name of the", "generate a title", "concise title",
)
_EPHEMERAL_KEYWORDS_LONG = (
    "suggest 3-5 relevant follow-up questions",
    "suggest 3-5 relevant follow-up",
    "user might naturally ask next in this conversation",
    "questions or prompts that the user might naturally ask",
    "summarizing the chat history",
    "summarize the chat",
    "summarize the conversation",
    "summarize the history",
    "summary of the chat",
    "summary of the conversation",
    "summary of the history",
    "concise, 3-5 word title",
    "concise title with an emoji",
    "emoji summarizing the chat",
    "generate a concise, 3-5 word title",
    "generate a concise title with an emoji",
    'json format: { "title":',
    'json format: {"title":',
    "guidelines:\n- the title should clearly represent",
)

def _is_ephemeral(messages: list) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if last.role != "user":
        return False
    c = last.content.lower()
    has_kw = any(kw in c for kw in _EPHEMERAL_KEYWORDS)
    has_history = any(m in c for m in (
        "assistant:", "assistant\n", "<|im_start|>assistant",
        "bot:", "ai:", "response:"
    ))
    if len(messages) == 1 and has_kw and has_history:
        return True
    if len(last.content) < 500 and has_kw:
        return True
    if any(kw in last.content.lower() for kw in _EPHEMERAL_KEYWORDS_LONG):
        return True
    return False

def _truncate_message_content(content: str) -> str:
    """Truncate long context bodies within messages to keep prompts small."""
    if len(content) <= 2000:
        return content
    c_lower = content.lower()
    for marker in ("chat history:", "conversation history:", "history:", "messages:", "context:", "document:", "text:"):
        idx = c_lower.find(marker)
        if idx != -1:
            prefix = content[:idx + len(marker)]
            body = content[idx + len(marker):]
            if len(body) > 1500:
                body = body[:750] + "\n...[truncated]...\n" + body[-750:]
            return prefix + body
    # Fallback: middle-truncate
    return content[:1000] + "\n...[truncated]...\n" + content[-1000:]

def _optimize_messages_for_ephemeral(messages: list) -> list:
    """Truncate long documents embedded in the history of title/follow-up requests."""
    optimized = []
    for msg in messages:
        if msg is messages[-1]:
            # Keep the final instruction prompt completely intact
            optimized.append(msg)
        else:
            optimized.append(ChatMessage(role=msg.role, content=_truncate_message_content(msg.content)))
    return optimized
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = list(request.messages) if request.messages else []

    # Detect and truncate ephemeral title/summary/follow-up requests (Open WebUI sends these automatically)
    is_ephemeral = _is_ephemeral(messages)
    if is_ephemeral:
        print(f"[Gateway] Detected ephemeral request ({sum(len(m.content) for m in messages)} chars total). Optimizing/Truncating.")
        messages = _optimize_messages_for_ephemeral(messages)
        wrapper = ephemeral_wrapper
        lock = ephemeral_lock
        ephemeral_wrapper.ensure_alive()
    else:
        wrapper = main_wrapper
        lock = main_lock

    if not messages:
        user_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    else:
        user_prompt = format_messages_as_chat(messages)

    max_tokens   = request.max_tokens or 16384
    request_id   = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # ── Session ID: derive from conversation history prefix (ACTIVE_RUNTIME style) ────────────
    # Use first user message + number of turns to create a stable session key.
    # Ephemeral requests always get a fresh session so they don't pollute the cache.
    import hashlib
    if is_ephemeral or not messages:
        session_id = "ephemeral"
    else:
        # Key on the first user message content (stable across turns of the same conversation)
        first_user = next((m.content for m in messages if m.role == "user"), "")
        session_id = hashlib.md5(first_user[:200].encode()).hexdigest()[:16]

    # ── Invalidate previous active session if switching conversation chats ──────────────────
    if not is_ephemeral:
        if wrapper.active_session_id and wrapper.active_session_id != session_id:
            print(f"[Gateway] Main session changed from {wrapper.active_session_id} to {session_id}. Invalidating previous cache.")
            wrapper._clear_session(wrapper.active_session_id)
        wrapper.active_session_id = session_id

    # ── Prefix match (ACTIVE_RUNTIME batch_engine.submit logic) ─────────────────────
    # Check if the current prompt starts with the prompt sent last turn.
    # Open WebUI sends the FULL conversation each turn, so turn 2's prompt IS
    # turn 1's prompt with the assistant reply and new user message appended.
    # We compare the full prev_prompt (not a truncated prefix) for maximum accuracy.
    cached_len = 0
    if not is_ephemeral and session_id in wrapper.session_cached_len:
        prev_cached  = wrapper.session_cached_len[session_id]
        if prev_cached > 0:
            # Pass previous cached length to C++ binary for token-level mismatch verification
            cached_len = prev_cached
            print(f"[Gateway] Session {session_id}: passing {cached_len} cached KV tokens to C++ binary for token-level verification.")


    # ── Streaming path ────────────────────────────────────────────────
    async def stream_generator():
        nonlocal cached_len
        async with lock:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def producer():
                wrapper.query_stream_into_queue(
                    user_prompt, max_tokens, queue, loop,
                    cached_len=cached_len, session_id=session_id
                )

            threading.Thread(target=producer, daemon=True).start()

            # ── SSE Keepalive: prevent HTTP connection timeout during prefill ──
            KEEPALIVE_S = 3.0
            prefill_done = False
            accumulated_response = []

            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except asyncio.TimeoutError:
                    if not prefill_done:
                        yield b": ping\n\n"
                    continue

                if item is None:
                    break

                if isinstance(item, dict) and "error" in item:
                    err_chunk = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created_time, "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": f"\n[Error: {item['error']}]"}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\n".encode()
                    wrapper._clear_session(session_id)
                    break

                if isinstance(item, dict) and item.get("prefill_done"):
                    prefill_done = True
                    text = item.get("text", "")
                    if not text:
                        continue

                # Update session cache after binary reports new KV pool size
                if isinstance(item, dict) and "cached_len" in item:
                    new_cached = item["cached_len"]
                    if not is_ephemeral:
                        wrapper.session_cached_len[session_id]  = new_cached
                        wrapper.session_prompt_text[session_id] = user_prompt
                        print(f"[Gateway] Session {session_id}: updated KV cache to {new_cached} tokens.")
                    continue  # don't yield this meta item

                text = item.get("text", "") if isinstance(item, dict) else ""
                if not text:
                    continue

                accumulated_response.append(text)

                chunk_data = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()

            # Update final prompt text to include the full response
            if not is_ephemeral:
                full_reply = "".join(accumulated_response)
                wrapper.session_prompt_text[session_id] = user_prompt + full_reply

            # Final done chunk
            done_chunk = {
                "id": request_id, "object": "chat.completion.chunk",
                "created": created_time, "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(done_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    # ── Non-streaming path ────────────────────────────────────────────────────
    async def collect_full():
        async with lock:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def producer():
                wrapper.query_stream_into_queue(
                    user_prompt, max_tokens, queue, loop,
                    cached_len=cached_len, session_id=session_id
                )

            threading.Thread(target=producer, daemon=True).start()

            parts = []
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    if "cached_len" in item:
                        new_cached = item["cached_len"]
                        if not is_ephemeral:
                            wrapper.session_cached_len[session_id]  = new_cached
                            wrapper.session_prompt_text[session_id] = user_prompt
                            print(f"[Gateway] Session {session_id}: updated KV cache to {new_cached} tokens.")
                        continue
                    t = item.get("text", "")
                    if t:
                        parts.append(t)
                else:
                    parts.append(str(item))
            full_response = "".join(parts)
            normalized_response = _normalize_references(full_response)
            if not is_ephemeral:
                wrapper.session_prompt_text[session_id] = user_prompt + normalized_response
            return normalized_response

    if request.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full_response = await collect_full()
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_response},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    model_path = os.getenv("DIFFKV_MODEL_PATH", MODEL_PATH_DEFAULT)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_id   = f"diffkv-{model_name}"
    return {
        "object": "list",
        "data": [{
            "id": model_id, "name": model_id,
            "object": "model", "created": int(time.time()),
            "owned_by": "differential-kv"
        }]
    }

@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok"}


if __name__ == '__main__':
    import argparse
    import uvicorn
    import sys

    _runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _runtime_dir not in sys.path:
        sys.path.insert(0, _runtime_dir)

    parser = argparse.ArgumentParser(description="DiffKV C++ Native API Server")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--preset', type=str, choices=['low', 'mid', 'high'], default='mid')
    parser.add_argument('--max-tokens', type=int, default=16384)
    args = parser.parse_args()

    if args.preset == 'low':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "256"
    elif args.preset == 'high':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "2048"
    else:
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "512"

    os.environ["DIFFKV_MAX_TOKENS"] = str(args.max_tokens)

    if "DIFFKV_USE_GPU" not in os.environ:
        os.environ["DIFFKV_USE_GPU"] = "1"

    if "DIFFKV_MPS_APPROXIMATE_ATTN" not in os.environ:
        os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"

    if "DIFFKV_MICRO_BLOCK_SIZE" not in os.environ:
        os.environ["DIFFKV_MICRO_BLOCK_SIZE"] = "64"

    if "DIFFKV_BINARY_PATH" not in os.environ:
        os.environ["DIFFKV_BINARY_PATH"] = BINARY_PATH_DEFAULT

    if "DIFFKV_MODEL_PATH" not in os.environ:
        model_arg = args.model
        if model_arg.endswith(".gguf") and os.path.exists(model_arg):
            os.environ["DIFFKV_MODEL_PATH"] = os.path.abspath(model_arg)
        elif "0.5b" in model_arg.lower():
            os.environ["DIFFKV_MODEL_PATH"] = _model_path("qwen2.5-0.5b-instruct.gguf")
        elif "1.5b" in model_arg.lower():
            q4_path = _model_path("qwen2.5-1.5b-instruct-q4_k_m.gguf")
            if os.path.exists(q4_path):
                os.environ["DIFFKV_MODEL_PATH"] = q4_path
            else:
                os.environ["DIFFKV_MODEL_PATH"] = _model_path("qwen2.5-1.5b-instruct-q8_0.gguf")
        else:
            os.environ["DIFFKV_MODEL_PATH"] = MODEL_PATH_DEFAULT

    print(f"[DiffKV Native Server] Starting:")
    print(f"  Model:      {os.environ.get('DIFFKV_MODEL_PATH')}")
    print(f"  Chunk size: {os.environ.get('DIFFKV_PREFILL_CHUNK_SIZE')}")
    print(f"  Max tokens: {os.environ.get('DIFFKV_MAX_TOKENS')}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
