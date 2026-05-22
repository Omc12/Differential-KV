import requests
import time
import json
import concurrent.futures

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def make_request(i):
    payload = {
        "model": "Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Write a detailed summary of machine learning for user {i}."}
        ],
        "max_tokens": 150,
        "stream": True
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    tokens = 0
    first_token_time = None
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['delta'].get('content', '')
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            tokens += 1
                except Exception as e:
                    pass
                    
    end_time = time.time()
    ttft = first_token_time - start_time if first_token_time else 0
    duration = end_time - start_time
    
    return {
        "id": i,
        "tokens": tokens,
        "ttft": ttft,
        "duration": duration,
        "tps": tokens / duration if duration > 0 else 0
    }

print("Sending concurrent requests to the REAL serving stack...")
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(make_request, i) for i in range(4)]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

print("\n--- RESULTS ---")
for r in results:
    print(f"User {r['id']}: {r['tokens']} tokens | TTFT {r['ttft']:.2f}s | {r['tps']:.2f} TPS")

# Check models
print("\nChecking /v1/models (if implemented) or just testing if gateway handles other routes")
try:
    resp = requests.get("http://localhost:8000/v1/sessions")
    print("Sessions:", resp.json())
except Exception as e:
    print(e)
