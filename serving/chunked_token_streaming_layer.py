import asyncio
import time
import logging
from typing import AsyncGenerator, Dict, List, Any

class ChunkedTokenStreamingLayer:
    """
    STAGE 2 CDBE: Chunked Token Streaming Layer.
    Amortizes Python/Network overhead by streaming token groups instead of single tokens,
    while maintaining "materially real" streaming.
    """
    def __init__(self, token_queue: asyncio.Queue, chunk_size: int = 4, timeout_ms: float = 50.0):
        self.token_queue = token_queue
        self.chunk_size = chunk_size
        self.timeout_ms = timeout_ms
        self.logger = logging.getLogger("CDBEStreaming")

    async def stream_generator(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yields chunks of tokens as they become available.
        """
        buffer = []
        last_yield_ts = time.time()
        
        while True:
            try:
                # Wait for at least one token
                # If we have something in the buffer, we use a timeout
                timeout = (self.timeout_ms / 1000.0) if buffer else None
                
                try:
                    payload = await asyncio.wait_for(self.token_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Timeout reached, yield what we have
                    if buffer:
                        yield self._create_chunk_payload(buffer)
                        buffer = []
                        last_yield_ts = time.time()
                    continue

                buffer.append(payload)
                
                # Check if buffer is full or session is finished
                is_final = payload.get("is_final", False)
                
                if len(buffer) >= self.chunk_size or is_final:
                    yield self._create_chunk_payload(buffer)
                    buffer = []
                    last_yield_ts = time.time()
                
                if is_final:
                    break
                    
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                break

    def _create_chunk_payload(self, buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines multiple token payloads into a single chunk."""
        if not buffer:
            return {}
            
        combined_text = "".join(p["token_text"] for p in buffer)
        last_payload = buffer[-1]
        
        return {
            "session_id": last_payload["session_id"],
            "token_text": combined_text,
            "token_count": len(buffer),
            "decode_complete_ts": last_payload["decode_complete_ts"],
            "is_final": last_payload["is_final"],
            "chunk_size": len(buffer)
        }
