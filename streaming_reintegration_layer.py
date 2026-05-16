import json
import time
import asyncio
from typing import AsyncGenerator, Dict, Any

class StreamingReintegrationLayer:
    """
    PSR System 3: Streaming Reintegration Layer.
    Ensures realistic streaming overhead, token flushing, and API serialization are measured.
    """
    def __init__(self):
        self.total_streaming_overhead_ms = 0.0
        self.chunk_count = 0

    async def wrap_stream(
        self, 
        raw_token_gen: AsyncGenerator[str, None], 
        request_id: str, 
        model: str
    ) -> AsyncGenerator[str, None]:
        """Wraps a raw token generator in OpenAI-compatible SSE format with measured overhead."""
        
        start_request = time.perf_counter()
        
        async for token in raw_token_gen:
            chunk_start = time.perf_counter()
            
            # Simulate real serialization overhead
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            
            # JSON serialization is a real CPU cost in high-concurrency serving
            serialized_data = json.dumps(data)
            payload = f"data: {serialized_data}\n\n"
            
            # Simulate network flush / event loop yield
            await asyncio.sleep(0) 
            
            chunk_end = time.perf_counter()
            self.total_streaming_overhead_ms += (chunk_end - chunk_start) * 1000
            self.chunk_count += 1
            
            yield payload

        # Final [DONE] signal
        yield "data: [DONE]\n\n"

    def get_streaming_metrics(self):
        return {
            "avg_serialization_overhead_ms": self.total_streaming_overhead_ms / self.chunk_count if self.chunk_count else 0,
            "total_chunks": self.chunk_count
        }

    async def flush_tokens(self, buffer: List[str]):
        """Simulates buffering and flushing tokens in chunks."""
        if not buffer:
            return ""
        
        flushed = "".join(buffer)
        buffer.clear()
        return flushed
