"""
DEPRECATED — DO NOT USE.

This file used a 4-layer randomly-initialised TransformerEncoder as the model.
It is NOT connected to any real LLM. Its outputs are meaningless random tokens.

The real serving entry point is:
    python run_dkv_webui_server.py

which uses LGSResolver + DKVHFWrapper (real HuggingFace model).
"""

raise RuntimeError(
    "api/openai_compatible_server.py is deprecated and must not be run. "
    "Use: python run_dkv_webui_server.py"
)

from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import torch

from serving.real_sparse_serving_runtime import RealSparseServingRuntime
from api.request_validation_layer import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice
)
from api.sparse_session_manager import SparseSessionManager

app = FastAPI(title="Differential KV OpenAI-Compatible API")
runtime = RealSparseServingRuntime()
session_manager = SparseSessionManager()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Extract prompt from messages
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role}: {msg.content}\n"
    prompt += "assistant: "

    if request.stream:
        return StreamingResponse(
            stream_generator(request, request_id, created_time, prompt),
            media_type="text/event-stream"
        )
    else:
        # Synchronous generation
        result = runtime.generate(prompt, max_new_tokens=request.max_tokens or 50)
        
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result["text"]),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(prompt) // 4, # Rough estimate for stub
                "completion_tokens": result["tokens_generated"],
                "total_tokens": (len(prompt) // 4) + result["tokens_generated"]
            }
        )
        return response

async def stream_generator(request, request_id, created_time, prompt):
    # In a real implementation, we would yield tokens as they are generated.
    # For the stub runtime, we'll simulate the streaming by breaking the generated text into chunks
    # but still doing the REAL compute first (as per mandatory rules).
    
    result = runtime.generate(prompt, max_new_tokens=request.max_tokens or 50)
    words = result["text"].split()
    
    for i, word in enumerate(words):
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta={"content": word + " "},
                    finish_reason=None if i < len(words) - 1 else "stop"
                )
            ]
        )
        yield f"data: {json.dumps(chunk.dict())}\n\n"
        await asyncio.sleep(0.01) # Small delay to simulate network/processing
        
    yield "data: [DONE]\n\n"

@app.get("/v1/sessions")
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}

@app.post("/v1/sessions")
async def create_session(config: Optional[Dict[str, Any]] = None):
    session_id = session_manager.create_session(config)
    return {"session_id": session_id}

import asyncio
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
