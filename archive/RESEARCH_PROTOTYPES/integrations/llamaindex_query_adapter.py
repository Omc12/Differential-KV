"""
integrations/llamaindex_query_adapter.py

LlamaIndex adapter for Differential KV.
Integrates sparse cognition with LlamaIndex retrieval pipelines.
"""

from typing import Any, Optional, Dict, List

try:
    from llama_index.core.llms import (
        CustomLLM,
        CompletionResponse,
        CompletionResponseGen,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_completion_callback
except ImportError:
    # Minimal fallback for testing environments without LlamaIndex
    class CustomLLM:
        def __init__(self, **kwargs): pass
    CompletionResponse = Any
    CompletionResponseGen = Any
    LLMMetadata = Any
    def llm_completion_callback():
        return lambda x: x

class DKVLlamaIndexAdapter(CustomLLM):
    """
    LlamaIndex adapter for Differential KV.
    """
    context_window: int = 16384
    num_output: int = 512
    model_name: str = "dkv"
    endpoint_url: str = "http://localhost:8000/v1/chat/completions"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        try:
            return LLMMetadata(
                context_window=self.context_window,
                num_output=self.num_output,
                model_name=self.model_name,
            )
        except:
            return {}

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Standard completion."""
        import requests
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        response = requests.post(self.endpoint_url, json=payload)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        
        try:
            return CompletionResponse(text=text)
        except:
            return {"text": text}

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion."""
        import requests
        import json
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        response = requests.post(self.endpoint_url, json=payload, stream=True)
        response.raise_for_status()

        def gen():
            text = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        content = line_str[6:].strip()
                        if content == "[DONE]":
                            break
                        data = json.loads(content)
                        delta = data["choices"][0]["delta"].get("content", "")
                        text += delta
                        try:
                            yield CompletionResponse(text=text, delta=delta)
                        except:
                            yield {"text": text, "delta": delta}
        return gen()

if __name__ == "__main__":
    adapter = DKVLlamaIndexAdapter()
    print("DKVLlamaIndexAdapter module loaded.")
