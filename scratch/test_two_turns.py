import urllib.request
import json
import time

def main():
    url = "http://localhost:9099/v1/chat/completions"
    
    # ── Turn 1 ──────────────────────────────────────────────────────────────
    print("=== Sending Turn 1 Request ===")
    messages = [
        {"role": "user", "content": "Hi, I have a cat named Whiskers. Remember his name."}
    ]
    payload = {
        "model": "diffkv-native-qwen2.5-0.5b-instruct",
        "messages": messages,
        "stream": False,
        "max_tokens": 128
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as response:
        dur = time.time() - start
        res_json = json.loads(response.read().decode('utf-8'))
        reply = res_json["choices"][0]["message"]["content"]
        print(f"Turn 1 Success in {dur:.2f}s")
        print(f"Assistant Reply: {reply}\n")
        
        # Append reply for turn 2
        messages.append({"role": "assistant", "content": reply})

    # ── Turn 2 ──────────────────────────────────────────────────────────────
    print("=== Sending Turn 2 Request (Continuation) ===")
    messages.append({"role": "user", "content": "What was the name of my cat?"})
    payload["messages"] = messages
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as response:
        dur = time.time() - start
        res_json = json.loads(response.read().decode('utf-8'))
        reply = res_json["choices"][0]["message"]["content"]
        print(f"Turn 2 Success in {dur:.2f}s")
        print(f"Assistant Reply: {reply}\n")

if __name__ == "__main__":
    main()
