import urllib.request
import json
import time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"

def post_request(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def main():
    with open("/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/scratch/berry_paper.txt", "r") as f:
        paper_text = f.read()

    session_id = "berry_single_turn_session"

    print("--- Running Single Turn with Paper + Question ---")
    payload = {
        "model": "diffkv-serving",
        "messages": [
            {"role": "user", "content": f"Here is the paper reference:\n\n{paper_text}\n\nQuestion: What are the codimensions of degeneracies for: (1) real symmetric hermitian matrices, (2) complex hermitian matrices, and (3) nonhermitian matrices? Compare their eigenvalue and eigenvector behavior around the degeneracy based on the paper."}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "session_id": session_id
    }
    t0 = time.time()
    try:
        ans = post_request(API_URL, payload)
        content = ans["choices"][0]["message"]["content"]
        print("Response:")
        print(content)
        print(f"Time: {time.time() - t0:.2f}s\n")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
