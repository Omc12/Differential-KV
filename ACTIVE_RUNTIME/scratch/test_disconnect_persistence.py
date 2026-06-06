import asyncio
import os
import sys
import torch
import json
import httpx
import time

# Set SRL environment variables at the very start
os.environ["DIFFKV_SRL_VERBOSE"] = "1"
os.environ["DIFFKV_SRL_THRESHOLD"] = "5"
os.environ["DIFFKV_SRL_K_MIN"] = "8"
os.environ["DIFFKV_SRL_K_MAX"] = "50"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

async def test_disconnect_persistence():
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from serving.batch_engine import ContinuousBatchEngine
    from serving.production_session_manager import ProductionSessionManager
    from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Initializing model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 16}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()

    session_manager = ProductionSessionManager(
        kv_manager=wrapper.manager,
        max_resident_sessions=2
    )

    gateway = OpenAICompatibleAPIGateway(resolver=engine, session_manager=session_manager)
    transport = httpx.ASGITransport(app=gateway.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Warm-up run to compile PyTorch MPS kernels
        print("Warm-up pass...")
        resp = await client.post("/v1/sessions")
        warmup_session_id = resp.json()["session_id"]
        
        warmup_payload = {
            "model": "diffkv-serving",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "max_tokens": 5,
            "session_id": warmup_session_id
        }
        await client.post("/v1/chat/completions", json=warmup_payload)
        await client.delete(f"/v1/sessions/{warmup_session_id}")
        print("Warm-up complete.")

        # ── Turn 1: Long prompt ──
        secret_info = "The secret code word is: ALBATROSS. Remember this secret word.\n\n"
        filler = (
            "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
            "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
            "than on classical computers. "
        )
        prompt = secret_info + (filler * 30)

        payload1 = {
            "model": "diffkv-serving",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "max_tokens": 10,
            "temperature": 0.0
        }

        print("\n--- Sending Turn 1 (Long Prompt) ---")
        t0 = time.perf_counter()
        
        session_id = None
        assistant_response = []
        
        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload1) as response:
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
        except Exception as e:
            print("Stream reading error:", e)

        duration1 = (time.perf_counter() - t0) * 1000
        assistant_text = "".join(assistant_response)
        print(f"Turn 1 completed in {duration1:.2f}ms")
        print(f"Generated text: {repr(assistant_text)}")

        # Wait a moment for uvicorn task teardown / async loops to settle
        await asyncio.sleep(0.5)

        # Print all active sessions and message histories to verify
        print("\n--- Verifying Active Sessions and Message Histories ---")
        active_sessions = list(session_manager.active_sessions.keys())
        print(f"Active sessions: {active_sessions}")
        assert len(active_sessions) == 1, "Session should not have been deleted!"
        
        matched_session_id = active_sessions[0]
        histories = session_manager.get_history(matched_session_id)
        print(f"Stored history for session {matched_session_id}: {histories}")
        assert len(histories) == 2, "Message history was not stored!"

        # ── Turn 2: Follow-up ──
        # Open WebUI sends the entire history in request.messages, without session_id
        payload2 = {
            "model": "diffkv-serving",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": "hi"}
            ],
            "stream": True,
            "max_tokens": 15,
            "temperature": 0.0
        }

        print("\n--- Sending Turn 2 (Follow-up 'hi') ---")
        t0 = time.perf_counter()
        assistant_response2 = []
        
        async with client.stream("POST", "/v1/chat/completions", json=payload2) as response:
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
                        assistant_response2.append(content)

        duration2 = (time.perf_counter() - t0) * 1000
        assistant_text2 = "".join(assistant_response2)
        print(f"Turn 2 completed in {duration2:.2f}ms")
        print(f"Generated text 2: {repr(assistant_text2)}")

        # Verify that Turn 2 was processed much faster
        print(f"\nTime comparison: Turn 1 = {duration1:.2f}ms, Turn 2 = {duration2:.2f}ms")

    await engine.stop()

if __name__ == "__main__":
    asyncio.run(test_disconnect_persistence())
