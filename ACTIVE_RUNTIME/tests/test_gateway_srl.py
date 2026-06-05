import asyncio
import os
import sys
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def run_gateway_srl_test():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager
    from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Initializing model {MODEL} for API gateway test...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2
    )

    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    
    # We will test using client or directly calling FastAPI endpoints via ASGI/httpx
    # Since TestClient supports lifespan when using a context manager:
    import httpx
    
    # We will run the event loop and start the ASGI app using an AsyncClient
    transport = httpx.ASGITransport(app=gateway.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res = await client.get("/health")
        print("Health check response:", res.json())
        assert res.status_code == 200
        
        # 2. Get models
        res = await client.get("/v1/models")
        print("Models response:", res.json())
        assert res.status_code == 200
        
        # 3. Create a session
        res = await client.post("/v1/sessions")
        session_data = res.json()
        print("Create session response:", session_data)
        session_id = session_data["session_id"]
        assert session_id is not None
        
        # 4. Check initial session SRL info (not built yet)
        res = await client.get(f"/v1/sessions/{session_id}/srl")
        srl_info = res.json()
        print("Initial SRL info response:", srl_info)
        assert srl_info["srl_built"] is False
        
        # 5. Send completion request with custom SRL configurations
        payload = {
            "model": "diffkv-serving",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please reply in exactly one word: Success."}
            ],
            "session_id": session_id,
            "srl_enabled": True,
            "srl_threshold": 10,  # low threshold to test custom setting
            "srl_k_min": 15,
            "srl_k_max": 150,
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        print("Sending chat completion request with custom SRL configs...")
        res = await client.post("/v1/chat/completions", json=payload)
        completion_data = res.json()
        print("Completion response:", completion_data)
        assert res.status_code == 200
        assert "choices" in completion_data
        
        # 6. Verify that SRL config was applied to the session
        res = await client.get(f"/v1/sessions/{session_id}/srl")
        srl_info_after = res.json()
        print("SRL info after completion:", srl_info_after)
        
        # Wait, since the prompt is short (only ~20 tokens), the number of blocks is very small (1 block).
        # The SRL index is built if we completed prefill compression.
        # Let's verify that the config values (k_min, k_max, routing_threshold) are correctly reflected!
        assert srl_info_after["k_min"] == 15
        assert srl_info_after["k_max"] == 150
        assert srl_info_after["routing_threshold"] == 10
        
        print("\n[PASS] Dynamic SRL configuration and endpoint successfully verified!")

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(run_gateway_srl_test())
