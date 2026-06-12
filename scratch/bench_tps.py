import urllib.request
import json
import time

url = "http://localhost:8000/v1/chat/completions"

prompt = "Explain the history of the internet in detail, highlighting the key milestones, technological breakthroughs, and social impact."

payload = {
    "model": "diffkv-native-qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": False
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Sending request for TPS benchmarking...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as response:
        dur = time.time() - start
        res_body = response.read().decode('utf-8')
        res_json = json.loads(res_body)
        
        content = res_json["choices"][0]["message"]["content"]
        # Token count is approximately 1.3 * word count or we can use the exact usage from API response
        usage = res_json.get("usage", {})
        if usage is None:
            usage = {}
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens == 0:
            completion_tokens = int(len(content.split()) * 1.3)
        prompt_tokens = usage.get("prompt_tokens", 0)
        
        tps = completion_tokens / dur
        print(f"\nSuccess!")
        print(f"Prompt tokens:     {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(f"Elapsed time:      {dur:.2f}s")
        print(f"Decode TPS:        {tps:.2f} tokens/sec")
        print("\nGenerated Text Preview:")
        print(content[:300] + "...")
except Exception as e:
    print(f"Request failed: {e}")
