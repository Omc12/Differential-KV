import urllib.request
import urllib.error
import json
import time

url = "http://localhost:8000/v1/chat/completions"

# Turn 1
messages = [{"role": "user", "content": "Hi, my favorite color is blue."}]
payload = {
    "model": "qwen2.5-0.5b-instruct",
    "messages": messages,
    "stream": False,
    "max_tokens": 100
}

print("=== Sending Turn 1 Request ===")
start = time.time()
try:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        res_body = response.read().decode('utf-8')
        res_json = json.loads(res_body)
        reply = res_json["choices"][0]["message"]["content"]
        print(f"Turn 1 Success in {time.time() - start:.2f}s")
        print(f"Reply: {reply}\n")
        
        # Append assistant's response to messages for Turn 2
        messages.append({"role": "assistant", "content": reply})
except Exception as e:
    print(f"Turn 1 Request failed: {e}")
    exit(1)

# Turn 2
messages.append({"role": "user", "content": "What is my favorite color?"})
payload["messages"] = messages

print("=== Sending Turn 2 Request ===")
start = time.time()
try:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        res_body = response.read().decode('utf-8')
        res_json = json.loads(res_body)
        reply = res_json["choices"][0]["message"]["content"]
        print(f"Turn 2 Success in {time.time() - start:.2f}s")
        print(f"Reply: {reply}\n")
except Exception as e:
    print(f"Turn 2 Request failed: {e}")
