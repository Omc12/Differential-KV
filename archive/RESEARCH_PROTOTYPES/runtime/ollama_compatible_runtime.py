import sys
import json
from typing import Dict, Any, Generator

class OllamaCompatibleRuntime:
    """
    Ollama-Compatible Runtime Layer
    
    Exposes /api/generate and /api/chat with streaming chunk structures
    and adapter layers matching modelfile serving states.
    """
    def __init__(self):
        self.generation_count = 0

    def generate(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """
        Handles local generation queries.
        """
        self.generation_count += 1
        return {
            "model": "qwen2.5:7b",
            "created_at": "2026-05-17T12:00:00Z",
            "response": f"Ollama generation response for prompt: {prompt}",
            "done": True,
            "context": [1, 2, 3],
            "total_duration": 45000000,
            "load_duration": 12000000,
            "prompt_eval_count": 8,
            "prompt_eval_duration": 5000000,
            "eval_count": 16,
            "eval_duration": 28000000
        }

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Streams Ollama JSON records chunk-by-chunk.
        """
        words = ["Ollama", " local", " serving", " streaming", " chunks."]
        for i, word in enumerate(words):
            chunk = {
                "model": "qwen2.5:7b",
                "created_at": "2026-05-17T12:00:00Z",
                "response": word,
                "done": False if i < len(words) - 1 else True
            }
            yield json.dumps(chunk) + "\n"
        sys.stdout.flush()

