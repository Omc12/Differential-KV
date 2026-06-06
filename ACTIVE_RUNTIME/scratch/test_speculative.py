import asyncio
import os
import sys
import torch
import json
import httpx
import time

# Set HF token from user
os.environ["HF_TOKEN"] = "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"
os.environ["DIFFKV_TELEMETRY"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_speculative_decoding():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager
    from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

    MAIN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Initializing main model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    main_wrapper = DiffKVHFWrapper(MAIN_MODEL, config={"rank": 16}, device=device)
    
    print("Initializing draft model...")
    draft_wrapper = DiffKVHFWrapper(DRAFT_MODEL, config={"rank": 16}, device=device)
    
    print("Starting continuous batching engine with speculative decoding...")
    engine = ContinuousBatchEngine(main_wrapper, max_batch_size=1, draft_wrapper=draft_wrapper)
    engine.start()

    session_manager = ProductionSessionManager(
        kv_manager=main_wrapper.manager,
        max_resident_sessions=2
    )

    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    transport = httpx.ASGITransport(app=gateway.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("\n--- Sending request ---")
        prompt = "Explain in one sentence what quantum computing is."
        payload = {
            "model": "diffkv-serving",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 30,
            "temperature": 0.0
        }
        
        t0 = time.perf_counter()
        assistant_response = []
        
        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        data = json.loads(data_str)
                        if "choices" in data:
                            content = data["choices"][0]["delta"].get("content", "")
                            assistant_response.append(content)
                            print(content, end="", flush=True)
        except Exception as e:
            print("\nStream reading error:", e)

        print()
        duration = (time.perf_counter() - t0) * 1000
        assistant_text = "".join(assistant_response)
        print(f"\nSpeculative decoding completed in {duration:.2f}ms")
        print(f"Generated text: {repr(assistant_text)}")

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_speculative_decoding())
