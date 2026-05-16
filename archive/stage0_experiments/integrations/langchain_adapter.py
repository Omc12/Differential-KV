from typing import Any, List, Optional
import requests

class DiffKVLangChainAdapter:
    """
    A simple adapter for LangChain-like usage of Differential KV.
    Points to the OpenAI-compatible endpoint.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "diff-kv"):
        self.base_url = base_url
        self.model = model

    def predict(self, text: str, max_tokens: int = 100) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "stream": False
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream(self, text: str, max_tokens: int = 100):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "stream": True
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    import json
                    data = json.loads(data_str)
                    yield data["choices"][0]["delta"].get("content", "")

if __name__ == "__main__":
    # Example usage (assuming server is running)
    # adapter = DiffKVLangChainAdapter()
    # print(adapter.predict("Explain quantum computing."))
    pass
