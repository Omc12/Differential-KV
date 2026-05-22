import json
import logging
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional

class WebUIStreamingIntegrationLayer:
    """
    OIS Phase 40.1: WebUI Streaming Integration Layer.
    Supports live token streaming and chat session continuity.
    """
    def __init__(self):
        self.logger = logging.getLogger("WebUIStreaming")

    async def stream_tokens(self, generator: AsyncGenerator[str, None], session_id: str) -> AsyncGenerator[str, None]:
        """
        Wraps a token generator for WebUI-compatible streaming.
        Ensures chunks are properly buffered and formatted.
        """
        self.logger.info(f"Starting stream for session: {session_id}")
        try:
            async for token in generator:
                # Format as OpenAI-style streaming chunk
                chunk = {
                    "id": f"chatcmpl-{session_id}",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "diffkv-production",
                    "choices": [{
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk
            final_chunk = {
                "id": f"chatcmpl-{session_id}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "diffkv-production",
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            self.logger.warning(f"Stream cancelled for session: {session_id}")
            raise
        except Exception as e:
            self.logger.error(f"Streaming error in session {session_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    def format_websocket_message(self, message_type: str, data: Dict[str, Any]) -> str:
        """Formats data for websocket delivery."""
        return json.dumps({
            "type": message_type,
            "payload": data
        })
