"""
integrations/langchain_sparse_connector.py

LangChain connector for Differential KV.
Inherits from LangChain base classes to allow usage in chains and agents.
"""

from typing import Any, List, Optional, Dict, Iterator
import requests

# We use a fallback if langchain is not installed to avoid import errors
try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    # Minimal fallback for testing environments without LangChain
    class LLM:
        def __init__(self, **kwargs): pass
    CallbackManagerForLLMRun = Any

class DiffKVSparseLLM(LLM):
    """
    Differential KV Sparse LLM for LangChain.
    """
    endpoint_url: str = "http://localhost:8000/v1/chat/completions"
    model_name: str = "diff-kv"
    temperature: float = 0.7
    max_tokens: int = 256
    
    @property
    def _llm_type(self) -> str:
        return "differential_kv"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(self.endpoint_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream the LLM response."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        response = requests.post(self.endpoint_url, json=payload, stream=True)
        response.raise_for_status()
        
        import json
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    content = line_str[6:].strip()
                    if content == "[DONE]":
                        break
                    data = json.loads(content)
                    chunk = data["choices"][0]["delta"].get("content", "")
                    if chunk:
                        yield chunk
                        if run_manager:
                            run_manager.on_llm_new_token(chunk)

if __name__ == "__main__":
    llm = DiffKVSparseLLM()
    print("DiffKVSparseLLM connector initialized.")
