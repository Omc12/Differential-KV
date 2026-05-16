import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Persistent Reasoning Stream")
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New reasoning stream client connected.")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("Client disconnected from stream.")

    async def broadcast_reasoning_step(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/stream/reasoning/{session_id}")
async def reasoning_stream(websocket: WebSocket, session_id: str):
    """
    Provides a real-time gRPC/WebSocket stream of the agent's internal
    reasoning trajectory, latent drift, and manifold navigation events.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive commands from client (e.g. steer attention)
            data = await websocket.receive_text()
            logger.info(f"Received from {session_id}: {data}")
            
            # Simulate streaming reasoning output
            await websocket.send_text(json.dumps({
                "session": session_id,
                "event": "manifold_jump",
                "target_attractor": "coding_logic",
                "confidence": 0.94
            }))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
