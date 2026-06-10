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

BINARY_PATH = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
MODEL_PATH = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

class SubprocessWrapper:
    def __init__(self):
        self.process = None

    def start(self):
        if self.process is not None:
            self.process.terminate()
        self.process = subprocess.Popen(
            [BINARY_PATH, MODEL_PATH, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Read until we see the first "__READY__"
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Subprocess failed to start")
            if "__READY__" in line:
                break
        print("[Server] C++ Native process started and ready.")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def query_stream(self, prompt: str):
        # Replace newlines and carriage returns with spaces to avoid breaking stdin
        single_line_prompt = prompt.replace("\n", " ").replace("\r", " ")
        self.process.stdin.write(single_line_prompt + "\n")
        self.process.stdin.flush()

        # 1. Read until we see "__RESPONSE__"
        response_started = False
        buffer = ""
        while not response_started:
            char = self.process.stdout.read(1)
            if not char:
                raise RuntimeError("Subprocess exited unexpectedly")
            buffer += char
            if "__RESPONSE__" in buffer:
                response_started = True
                buffer = "" # Discard everything before and including __RESPONSE__

        # 2. Yield any remaining chars in the buffer (should be empty now)
        if buffer:
            yield buffer
            buffer = ""

        # 3. Stream from stdout file descriptor in real-time
        fd = self.process.stdout.fileno()
        accumulated = ""
        while True:
            r, _, _ = select.select([fd], [], [], 0.05)
            if fd in r:
                chunk = os.read(fd, 4096).decode("utf-8", errors="ignore")
                if not chunk:
                    break
                accumulated += chunk
                if "__FINISH__\n" in accumulated:
                    final_part = accumulated.split("__FINISH__\n")[0]
                    yield final_part
                    break
                else:
                    if len(accumulated) > 12:
                        yield accumulated[:-12]
                        accumulated = accumulated[-12:]
            else:
                # Sleep briefly to yield CPU
                time.sleep(0.005)

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
    # Extract last user message content
    user_prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_prompt = msg.content
            break
    if not user_prompt:
        user_prompt = "Hello"

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
