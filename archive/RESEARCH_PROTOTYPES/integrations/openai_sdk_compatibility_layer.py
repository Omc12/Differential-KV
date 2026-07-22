"""
integrations/openai_sdk_compatibility_layer.py

Validation and normalization layer for OpenAI SDK interoperability.
Ensures the Differential KV gateway works seamlessly with the official OpenAI client.
"""

import json
from typing import Dict, Any, List, Optional

class OpenAISDKCompatibilityLayer:
    """
    Verifies that Differential KV serving endpoints correctly handle
    standard OpenAI SDK requests and return compliant responses.
    """
    def __init__(self, api_base_url: str = "http://localhost:8000/v1"):
        self.api_base_url = api_base_url

    def validate_request_format(self, request_payload: Dict[str, Any]) -> bool:
        """
        Validates that a request from the OpenAI SDK is correctly structured.
        """
        required_fields = ["model", "messages"]
        for field in required_fields:
            if field not in request_payload:
                return False
        return True

    def normalize_response(self, raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the internal response from Differential KV is perfectly
        aligned with OpenAI's 'chat.completion' schema.
        """
        # Differential KV specific metadata can be stripped or moved to 'usage'
        normalized = {
            "id": raw_response.get("id", "chatcmpl-unknown"),
            "object": "chat.completion",
            "created": raw_response.get("created", 0),
            "model": raw_response.get("model", "dkv"),
            "choices": raw_response.get("choices", []),
            "usage": raw_response.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }
        return normalized

    def verify_streaming_chunk(self, chunk_data: str) -> bool:
        """
        Verifies that a streaming SSE chunk is valid JSON and follows OpenAI format.
        """
        if chunk_data.startswith("data: "):
            content = chunk_data[6:].strip()
            if content == "[DONE]":
                return True
            try:
                data = json.loads(content)
                return "choices" in data and "delta" in data["choices"][0]
            except json.JSONDecodeError:
                return False
        return False

if __name__ == "__main__":
    layer = OpenAISDKCompatibilityLayer()
    test_req = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]}
    print(f"Request Valid: {layer.validate_request_format(test_req)}")
