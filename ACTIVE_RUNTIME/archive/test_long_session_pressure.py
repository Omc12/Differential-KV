import asyncio
import aiohttp
import time
import json
import sys

print("=== PHASE 5: LONG SESSION PRESSURE TEST (MULTI-USER) ===")

API_URL = "http://localhost:8080/v1/chat/completions"

async def chat_session(session_id: int, num_turns: int):
    async with aiohttp.ClientSession() as session:
        messages = [
            {"role": "system", "content": "You are a highly intelligent physics assistant."}
        ]
        
        for turn in range(num_turns):
            prompt = f"Explain the concept of entropy in thermodynamics. Detail its mathematical formulation and give {turn+1} real-world examples."
            messages.append({"role": "user", "content": prompt})
            
            print(f"[User {session_id}] Sending Turn {turn+1} (History: {len(messages)} msgs)...")
            t0 = time.time()
            
            async with session.post(API_URL, json={
                "model": "DifferentialKV",
                "messages": messages,
                "stream": False,
                "max_tokens": 150
            }) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"[User {session_id}] Error {resp.status}: {text}")
                    break
                    
                data = await resp.json()
                reply = data['choices'][0]['message']['content']
                t1 = time.time()
                
                print(f"[User {session_id}] Turn {turn+1} Done in {t1-t0:.2f}s. Reply len: {len(reply)} chars")
                messages.append({"role": "assistant", "content": reply})
                
                # Sleep a bit to simulate human reading time
                await asyncio.sleep(1.0)

async def main():
    NUM_USERS = 4
    NUM_TURNS = 5
    
    tasks = []
    t_start = time.time()
    
    for i in range(NUM_USERS):
        tasks.append(chat_session(i+1, NUM_TURNS))
        
    await asyncio.gather(*tasks)
    
    t_end = time.time()
    print(f"\nPressure Test Complete in {t_end-t_start:.2f} seconds.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
