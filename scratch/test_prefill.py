import urllib.request
import json
import time

url = "http://127.0.0.1:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Place the crucial information at the very beginning of the prompt
secret_info = "The secret code word is: ALBATROSS. Remember this secret word.\n\n"

filler = (
    "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
    "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
    "than on classical computers. The field of quantum computing includes hardware research and "
    "application development. Quantum computers are able to solve certain classes of problems "
    "much faster than classical computers by taking advantage of quantum mechanical effects, "
    "such as superposition and quantum entanglement. "
)

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    + secret_info
    + (filler * 12) + "\n\n"  # Around 800 tokens
    + "Question: What is the secret code word? Answer in exactly one word.<|im_end|>\n"
    + "<|im_start|>assistant\n"
)

data = {
    "model": "qwen2.5-0.5b-instruct.gguf",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": True  # Enable streaming
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)

print("Sending chat completions request to port 8001 (streaming mode)...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as response:
        print("Connected to stream. Receiving chunks:")
        first_token_time = None
        for line in response:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            if line_str.startswith("data: [DONE]"):
                print("\n[DONE]")
                break
            if line_str.startswith("data: "):
                try:
                    chunk = json.loads(line_str[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                            print(f"\nTime to first token: {first_token_time - t0:.2f} seconds.")
                        print(content, end="", flush=True)
                except Exception as ex:
                    print(f"\nError parsing chunk: {ex} (line: {line_str})")
        elapsed = time.time() - t0
        print(f"\nStreaming completed in {elapsed:.2f} seconds.")
except Exception as e:
    print(f"\nError during request: {e}")
