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

    session_id = "berry_test_session_123"

    print("--- Turn 1: Ingesting Berry paper ---")
    payload = {
        "model": "diffkv-serving",
        "messages": [
            {"role": "user", "content": f"Here is the paper reference:\n\n{paper_text}\n\nAcknowledge that you have received the paper by saying 'Paper received successfully.'"}
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "session_id": session_id
    }
    t0 = time.time()
    try:
        ans_1 = post_request(API_URL, payload)
        content_1 = ans_1["choices"][0]["message"]["content"]
        print("Response 1:", content_1)
        print(f"Time: {time.time() - t0:.2f}s\n")
    except Exception as e:
        print("Error on Turn 1:", e)
        return

    print("--- Turn 2: Querying codimension and eigenvalue behavior ---")
    payload_2 = {
        "model": "diffkv-serving",
        "messages": [
            {"role": "user", "content": f"Here is the paper reference:\n\n{paper_text}\n\nAcknowledge that you have received the paper by saying 'Paper received successfully.'"},
            {"role": "assistant", "content": content_1},
            {"role": "user", "content": "Question: What are the codimensions of degeneracies for: (1) real symmetric hermitian matrices, (2) complex hermitian matrices, and (3) nonhermitian matrices? Compare their eigenvalue and eigenvector behavior around the degeneracy based on the paper."}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "session_id": session_id
    }
    t0 = time.time()
    try:
        ans_2 = post_request(API_URL, payload_2)
        content_2 = ans_2["choices"][0]["message"]["content"]
        print("Response 2:")
        print(content_2)
        print(f"Time: {time.time() - t0:.2f}s\n")
    except Exception as e:
        print("Error on Turn 2:", e)

if __name__ == "__main__":
    main()
