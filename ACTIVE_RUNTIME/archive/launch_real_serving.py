import asyncio
import uvicorn
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway
from serving.production_session_manager import ProductionSessionManager
from fastapi.middleware.cors import CORSMiddleware

print("Initializing Model Wrapper (Qwen/Qwen2.5-0.5B-Instruct)...")
wrapper = DiffKVHFWrapper("Qwen/Qwen2.5-0.5B-Instruct", {"mode": "fp16", "block_size": 64, "rank": 16})
engine = ContinuousBatchEngine(wrapper, max_batch_size=8)
session_manager = ProductionSessionManager()

gateway = OpenAICompatibleAPIGateway(engine, session_manager=session_manager)

gateway.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifespan events handled above

if __name__ == "__main__":
    print("Starting Differential KV Real Serving Entrypoint...")
    uvicorn.run(gateway.app, host="0.0.0.0", port=8080)
