import os
import subprocess
import asyncio
import json
import uuid
import time
import select
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
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7

BINARY_PATH_DEFAULT = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
MODEL_PATH_DEFAULT = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"

class SubprocessWrapper:
    def __init__(self):
        self.process = None

    def start(self):
        if self.process is not None:
            self.process.terminate()
        
        binary_path = os.getenv("DIFFKV_BINARY_PATH", BINARY_PATH_DEFAULT)
        model_path = os.getenv("DIFFKV_MODEL_PATH", MODEL_PATH_DEFAULT)
        
        print(f"[Server] Launching C++ subprocess: {binary_path} {model_path} -")
        self.process = subprocess.Popen(
            [binary_path, model_path, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=False,  # binary mode; we decode manually with errors='replace'
            bufsize=0
        )
        
        # Read until we see the first "__READY__"
        while True:
            line_bytes = self.process.stdout.readline()
            if not line_bytes:
                raise RuntimeError("Subprocess failed to start")
            line = line_bytes.decode("utf-8", errors="replace")
            if "__READY__" in line:
                break
        print("[Server] C++ Native process started and ready.")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def _write_stdin(self, text: str):
        """Write text to subprocess stdin (binary mode)."""
        self.process.stdin.write(text.encode("utf-8"))
        self.process.stdin.flush()

    def _read_byte(self) -> str:
        """Read one byte from subprocess stdout and decode with replacement."""
        b = self.process.stdout.read(1)
        if not b:
            return ""
        return b.decode("utf-8", errors="replace")

    def query_stream(self, prompt: str):
        # Escape real newlines as \\n so the prompt can travel on a single stdin line.
        # The C++ binary decodes \\n back to real newlines before processing.
        single_line_prompt = prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
        self._write_stdin(single_line_prompt + "\n")

        # 1. Read until we see "__RESPONSE__"
        response_started = False
        buffer = ""
        while not response_started:
            char = self._read_byte()
            if not char:
                raise RuntimeError("Subprocess exited unexpectedly")
            buffer += char
            if "__RESPONSE__" in buffer:
                response_started = True
                buffer = ""  # Discard everything before and including __RESPONSE__

        # 2. Yield any remaining chars in the buffer (should be empty now)
        if buffer:
            yield buffer
            buffer = ""

        # 3. Stream from stdout in real-time
        accumulated = ""
        while True:
            char = self._read_byte()
            if not char:
                break
            accumulated += char
            if "__FINISH__\n" in accumulated:
                final_part = accumulated.split("__FINISH__\n")[0]
                yield final_part
                break
            else:
                if len(accumulated) > 12:
                    yield accumulated[:-12]
                    accumulated = accumulated[-12:]


def format_messages_as_chat(messages: list) -> str:
    """
    Build a full Qwen2.5 chat-template string from a list of ChatMessage objects.
    Includes system prompt and all user/assistant turns so the C++ binary has
    full conversation context without any redundant template wrapping.
    """
    result = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    for msg in messages:
        role = msg.role
        content = msg.content
        if role in ("system", "user", "assistant"):
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
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
    # Format the FULL conversation history with Qwen2.5 chat template.
    # The C++ binary detects the <|im_start|> prefix and skips its own template wrapping.
    if not request.messages:
        user_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    else:
        user_prompt = format_messages_as_chat(request.messages)


    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    async def stream_generator():
        # Ensure mutual exclusion on the subprocess
        async with lock:
            loop = asyncio.get_event_loop()
            queue = asyncio.Queue()

            def producer():
                try:
                    for token in wrapper.query_stream(user_prompt):
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                except Exception as e:
                    print(f"Error in producer: {e}")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            # Start producer thread
            import threading
            threading.Thread(target=producer, daemon=True).start()

            # Consumer loop
            while True:
                token = await queue.get()
                if token is None:
                    break
                if not token:
                    continue

                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            # Final done chunk
            done_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # Non-streaming mode: collect all tokens
        async with lock:
            def run_sync_all():
                return "".join(wrapper.query_stream(user_prompt))
            
            loop = asyncio.get_event_loop()
            full_response = await loop.run_in_executor(None, run_sync_all)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    model_path = os.getenv("DIFFKV_MODEL_PATH", MODEL_PATH_DEFAULT)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_id = f"diffkv-native-{model_name}"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "name": model_id,
            "object": "model",
            "created": int(time.time()),
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
    import os

    _runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _runtime_dir not in sys.path:
        sys.path.insert(0, _runtime_dir)

    parser = argparse.ArgumentParser(description="DiffKV C++ Native API Server")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--preset', type=str, choices=['low', 'mid', 'high'], default='mid')
    parser.add_argument('--serving-mode', type=str, choices=['lightweight', 'balanced', 'performance'], default='balanced')
    parser.add_argument('--micro-block-size', type=int, default=16)
    parser.add_argument('--gpu-budget-gb', type=float, default=2.0)
    parser.add_argument('--max-context-slots', type=int, default=512)
    args = parser.parse_args()

    # Determine preset based on args or serving mode
    preset = args.preset
    if args.serving_mode == 'lightweight':
        preset = 'low'
    elif args.serving_mode == 'performance':
        preset = 'high'

    # Set env variables
    os.environ["DIFFKV_PRESET"] = preset
    if preset == 'low':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "256"
    elif preset == 'high':
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "2048"
    else:
        os.environ["DIFFKV_PREFILL_CHUNK_SIZE"] = "512"

    os.environ["DIFFKV_MICRO_BLOCK_SIZE"] = str(args.micro_block_size)
    os.environ["DIFFKV_GPU_BUDGET_GB"] = str(args.gpu_budget_gb)
    os.environ["DIFFKV_MAX_CONTEXT_SLOTS"] = str(args.max_context_slots)

    if "DIFFKV_BINARY_PATH" not in os.environ:
        os.environ["DIFFKV_BINARY_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"

    # Resolve model path
    if "DIFFKV_MODEL_PATH" not in os.environ:
        model_arg = args.model
        if model_arg.endswith(".gguf") and os.path.exists(model_arg):
            os.environ["DIFFKV_MODEL_PATH"] = os.path.abspath(model_arg)
        elif "0.5b" in model_arg.lower():
            os.environ["DIFFKV_MODEL_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
        elif "1.5b" in model_arg.lower():
            os.environ["DIFFKV_MODEL_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"
        else:
            default_path = os.path.join("/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native", model_arg)
            if os.path.exists(default_path):
                os.environ["DIFFKV_MODEL_PATH"] = default_path
            else:
                # Fallback to default
                os.environ["DIFFKV_MODEL_PATH"] = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-1.5b-instruct-q8_0.gguf"

    print(f"[DiffKV Native Server] Starting server with configurations:")
    print(f"  - Model Path:         {os.environ.get('DIFFKV_MODEL_PATH')}")
    print(f"  - Preset:             {os.environ.get('DIFFKV_PRESET')}")
    print(f"  - Prefill Chunk Size: {os.environ.get('DIFFKV_PREFILL_CHUNK_SIZE')}")
    print(f"  - Max Context Slots:  {os.environ.get('DIFFKV_MAX_CONTEXT_SLOTS')}")
    print(f"  - Micro Block Size:   {os.environ.get('DIFFKV_MICRO_BLOCK_SIZE')}")
    print(f"  - GPU Budget (GB):    {os.environ.get('DIFFKV_GPU_BUDGET_GB')}")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)

