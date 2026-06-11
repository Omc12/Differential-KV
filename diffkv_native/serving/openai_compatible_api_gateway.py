import os
import subprocess
import asyncio
import json
import uuid
import time
import threading
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

BINARY_PATH_DEFAULT = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
MODEL_PATH_DEFAULT  = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

# Sentinel bytes
_SENTINEL_RESPONSE = b"__RESPONSE__"
_SENTINEL_FINISH   = b"__FINISH__"
_SENTINEL_READY    = b"__READY__"

class SubprocessWrapper:
    def __init__(self):
        self.process = None

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
        self.process = subprocess.Popen(
            [binary_path, model_path, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,   # inherit stderr so logs go to terminal
            text=False,
            bufsize=0,
        )

        # Drain stdout until __READY__
        buf = b""
        while True:
            b = self.process.stdout.read(1)
            if not b:
                raise RuntimeError("Subprocess failed to start (stdout closed)")
            buf += b
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

    def query_stream_into_queue(self, prompt: str, max_tokens: int, out_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """
        Runs in a background thread.
        Writes prompt to C++ stdin, reads stdout byte-by-byte.
        Puts decoded text chunks into out_queue.
        Puts None when done (signals end-of-stream).
        Puts {"error": msg} dict on fatal errors.

        Protocol:
          C++ stdout: ... __RESPONSE__\\n <tokens...> __FINISH__\\n
        """
        # Auto-restart if process has crashed between requests
        self.ensure_alive()

        # Escape newlines and backslashes so the prompt fits on one stdin line.
        # The C++ binary decodes \\n back to real newlines before tokenizing.
        single_line = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")

        try:
            self._write_stdin(single_line + "\n")
        except BrokenPipeError:
            # Process died mid-write — restart and retry once
            print("[Server] BrokenPipeError writing prompt — restarting C++ process...", flush=True)
            try:
                self.start()
                self._write_stdin(single_line + "\n")
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
                b = self.process.stdout.read(1)
                if not b:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"error": "process exited before __RESPONSE__"})
                    loop.call_soon_threadsafe(out_queue.put_nowait, None)
                    return
                buf += b
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
        if buf:
            text = buf.decode("utf-8", errors="replace")
            # Check if FINISH is already in this fragment
            if _SENTINEL_FINISH.decode() in text:
                final = text.split(_SENTINEL_FINISH.decode())[0]
                if final:
                    loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final})
                loop.call_soon_threadsafe(out_queue.put_nowait, None)
                return
            if text:
                loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})

        # ── Phase 2: stream tokens until __FINISH__ ──────────────────────────
        accumulated = b""
        try:
            while True:
                b = self.process.stdout.read(1)
                if not b:
                    # Process died — treat as finish
                    break
                accumulated += b
                finish_str = _SENTINEL_FINISH.decode()
                acc_str = accumulated.decode("utf-8", errors="replace")
                if finish_str in acc_str:
                    final_part = acc_str.split(finish_str)[0]
                    if final_part:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": final_part})
                    break
                # Flush safely: keep a tail buffer long enough to hold the sentinel
                tail_len = len(_SENTINEL_FINISH) + 4
                if len(accumulated) > tail_len:
                    safe = accumulated[:-tail_len]
                    remaining = accumulated[-tail_len:]
                    text = safe.decode("utf-8", errors="replace")
                    if text:
                        loop.call_soon_threadsafe(out_queue.put_nowait, {"text": text})
                    accumulated = remaining
        except Exception as e:
            loop.call_soon_threadsafe(out_queue.put_nowait, {"error": f"read error during generation: {e}"})

        loop.call_soon_threadsafe(out_queue.put_nowait, None)


def format_messages_as_chat(messages: list) -> str:
    """Build a full Qwen2.5 chat-template string from ChatMessage objects."""
    result = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for msg in messages:
        if msg.role in ("system", "user", "assistant"):
            result += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


# Initialize wrapper
wrapper = SubprocessWrapper()

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    wrapper.start()
    yield
    wrapper.stop()

app.router.lifespan_context = lifespan


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        user_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    else:
        user_prompt = format_messages_as_chat(request.messages)

    max_tokens   = request.max_tokens or 512
    request_id   = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # ── Streaming path ────────────────────────────────────────────────────────
    async def stream_generator():
        async with lock:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            # Set max_tokens env before the thread reads it (C++ already running,
            # so pass via a wrapper attribute used at query time instead)
            # We do it by setting DIFFKV_MAX_TOKENS in the environment that the
            # *next* C++ restart would see. For the current session, the C++
            # binary uses whatever was set at start time. Best effort: we restart
            # the C++ binary if max_tokens changed significantly (TODO).
            # For now: just run — the default is 512 in C++.

            def producer():
                wrapper.query_stream_into_queue(user_prompt, max_tokens, queue, loop)

            threading.Thread(target=producer, daemon=True).start()

            # ── SSE Keepalive: prevent HTTP connection timeout during prefill ──
            # Matching ACTIVE_RUNTIME: send ": ping\n\n" every 3s while waiting
            # for __RESPONSE__. Open WebUI drops the connection after ~30s silence.
            KEEPALIVE_S = 3.0
            prefill_done = False

            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
                except asyncio.TimeoutError:
                    if not prefill_done:
                        # Still in prefill — send keepalive comment
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
                    break

                if isinstance(item, dict) and item.get("prefill_done"):
                    prefill_done = True
                    # Don't yield anything — just mark that prefill is done
                    text = item.get("text", "")
                    if not text:
                        continue

                text = item.get("text", "") if isinstance(item, dict) else ""
                if not text:
                    continue

                chunk_data = {
                    "id": request_id, "object": "chat.completion.chunk",
                    "created": created_time, "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()

            # Final done chunks
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
                wrapper.query_stream_into_queue(user_prompt, max_tokens, queue, loop)

            threading.Thread(target=producer, daemon=True).start()

            parts = []
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if t:
                        parts.append(t)
                else:
                    parts.append(str(item))
            return "".join(parts)

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
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()

    if args.preset == 'low':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "256"
    elif args.preset == 'high':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "2048"
    else:
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "512"

    os.environ["DIFFKV_MAX_TOKENS"] = str(args.max_tokens)

    if "DIFFKV_BINARY_PATH" not in os.environ:
        os.environ["DIFFKV_BINARY_PATH"] = BINARY_PATH_DEFAULT

    if "DIFFKV_MODEL_PATH" not in os.environ:
        model_arg = args.model
        if model_arg.endswith(".gguf") and os.path.exists(model_arg):
            os.environ["DIFFKV_MODEL_PATH"] = os.path.abspath(model_arg)
        elif "0.5b" in model_arg.lower():
            os.environ["DIFFKV_MODEL_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
        elif "1.5b" in model_arg.lower():
            os.environ["DIFFKV_MODEL_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
        else:
            os.environ["DIFFKV_MODEL_PATH"] = MODEL_PATH_DEFAULT

    print(f"[DiffKV Native Server] Starting:")
    print(f"  Model:      {os.environ.get('DIFFKV_MODEL_PATH')}")
    print(f"  Chunk size: {os.environ.get('DIFFKV_PREFILL_CHUNK_SIZE')}")
    print(f"  Max tokens: {os.environ.get('DIFFKV_MAX_TOKENS')}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
