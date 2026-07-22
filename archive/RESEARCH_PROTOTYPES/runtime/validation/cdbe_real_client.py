import asyncio
import aiohttp
import json
import time
import sys
import random

async def run_client_session(session_id, prompt, max_tokens=512):
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "dkv-qwen2.5-7b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "session_id": session_id
    }
    
    tokens_received = 0
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"[{session_id}] Error: {response.status}")
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    tokens_received += 1
                                    # Print a dot for every token from ANY session to show live activity
                                    sys.stdout.write(".")
                                    sys.stdout.flush()
                                    if tokens_received % 50 == 0:
                                        sys.stdout.write(f"\n[{session_id}] ")
                        except:
                            continue
                            
    except Exception as e:
        print(f"[{session_id}] Connection Failed: {e}")

    duration = time.time() - start_time
    return tokens_received, duration

async def run_load_test():
    print("\n" + "="*60)
    print("CDBE REAL LOAD CLIENT: 16 CONCURRENT SESSIONS")
    print("="*60)
    
    concurrency = 4
    long_prompt = "Tell me a very long and detailed story about the future of artificial intelligence and its impact on space exploration. " * 20
    
    print(f"[*] Launching {concurrency} concurrent requests...")
    
    tasks = []
    for i in range(concurrency):
        tasks.append(run_client_session(f"session-{i}", long_prompt))
        # Small stagger to see queue depth changes
        await asyncio.sleep(random.uniform(0.1, 0.5))
    
    print(f"[*] All {concurrency} sessions active. Streaming Session 0 output below:\n")
    print("-" * 30)
    
    results = await asyncio.gather(*tasks)
    
    print("\n" + "-" * 30)
    print("[*] All sessions completed.")
    
    total_tokens = sum(r[0] for r in results if r)
    avg_duration = sum(r[1] for r in results if r) / concurrency
    
    print(f"[*] Total Tokens: {total_tokens}")
    print(f"[*] Avg Duration: {avg_duration:.2f}s")

if __name__ == "__main__":
    # Wait a bit for server to be ready
    time.sleep(2)
    asyncio.run(run_load_test())
