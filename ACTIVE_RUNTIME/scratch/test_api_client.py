import urllib.request
import json
import time

url = "http://127.0.0.1:8000/v1/chat/completions"

# 1. Simple query
payload1 = {
    "model": "diffkv-serving",
    "messages": [
        {"role": "user", "content": "Hello! How are you?"}
    ],
    "temperature": 0.0,
    "max_tokens": 50,
    "stream": False
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload1).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Sending Turn 1 (Hello!)...")
t0 = time.time()
try:
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        body = response.read().decode("utf-8")
        print(f"Status code: {status_code}")
        print(f"Time taken: {time.time() - t0:.2f}s")
        print("Response:")
        print(json.dumps(json.loads(body), indent=2))
except Exception as e:
    print(f"Request failed: {e}")

