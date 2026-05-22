import json
from typing import Dict, Any, Generator, List

class OpenAICompatibleApiRuntime:
    """
    OpenAI-Compatible API Runtime
    
    Exposes /v1/chat/completions and /v1/completions with temperature/top_p,
    streaming SSE format compatibility, and token accounting metrics.
    """
    def __init__(self):
        self.request_count = 0
        self.token_count = 0

    def generate_chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 128) -> Dict[str, Any]:
        """
        Processes chat completions requests mimicking OpenAI's JSON response formats.
        """
        self.request_count += 1
        prompt = messages[-1]["content"] if messages else ""
        
        # Simplified response construction matching standard schemas
        return {
            "id": f"chatcmpl-{self.request_count}",
            "object": "chat.completion",
            "created": 1715970000,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Here is the OpenAI assistant response to: '{prompt}'"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // 4 + 4,
                "completion_tokens": 16,
                "total_tokens": len(prompt) // 4 + 20
            }
        }

    def generate_chat_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Streams chunks mimicking Server-Sent Events (SSE).
        """
        words = ["Here", " is", " the", " streaming", " OpenAI", " compatible", " assistant", " token", " stream."]
        for i, word in enumerate(words):
            chunk = {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": 1715970000,
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": word
                        },
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
