import asyncio
import aiohttp
import time
import json

async def fetch(session, idx, prompt):
    start = time.time()
    payload = {
        "model": "dkv-serving",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    
    first_token_time = None
    tokens = 0
    text = ""
    
    try:
        async with session.post("http://127.0.0.1:8000/v1/chat/completions", json=payload) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        chunk = data["choices"][0]["delta"].get("content", "")
                        if chunk:
                            if first_token_time is None:
                                first_token_time = time.time()
                            text += chunk
                            # Approximate tokens
                            tokens += len(chunk) / 4
                    except Exception as e:
                        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Request {idx} failed: {e}")
        return None
                        
    end = time.time()
    ttft = first_token_time - start if first_token_time else 0
    return {
        "idx": idx,
        "ttft": ttft,
        "total_time": end - start,
        "tokens": int(tokens),
        "tps": int(tokens) / (end - start)
    }

async def main():
    prompts = [
        "Write a detailed essay about the Roman Empire.",
        "Explain the process of photosynthesis step by step.",
        "Write a short story about a time traveler.",
        "Describe the architecture of a modern CPU.",
        "What are the main differences between Python and C++?",
        "Explain quantum entanglement to a high schooler.",
        "Write a poem about the ocean.",
        "How does a database index work?"
    ]
    
    print(f"Firing {len(prompts)} concurrent requests...")
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
    end_time = time.time()
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("All requests failed.")
        return
        
    avg_ttft = sum(r["ttft"] for r in valid_results) / len(valid_results)
    total_tokens = sum(r["tokens"] for r in valid_results)
    total_time = end_time - start_time
    aggregate_tps = total_tokens / total_time
    
    print("\n--- REAL BATCHING RESULTS ---")
    print(f"Concurrent Requests: {len(valid_results)}")
    print(f"Average TTFT: {avg_ttft:.2f} s")
    print(f"Total Tokens: {total_tokens}")
    print(f"Wall-Clock Time: {total_time:.2f} s")
    print(f"Aggregate System TPS: {aggregate_tps:.2f} tokens/s")
    
    for r in valid_results:
        print(f"Req {r['idx']} | TTFT: {r['ttft']:.2f}s | TPS: {r['tps']:.2f} | Time: {r['total_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
