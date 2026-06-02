import urllib.request
import json
import time

API_URL = "http://127.0.0.1:8000/v1/chat/completions"

PAPER_ABSTRACT = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""

def post_request(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def test_flow():
    # 1. Turn 1: Send large prompt with temp=0.7, top_p=0.9, rep_penalty=1.15
    print("Sending Turn 1 (Sustainable AI paper abstract) with non-zero temp...")
    payload = {
        "model": "diffkv-serving",
        "messages": [
            {"role": "user", "content": f"Here is the paper abstract:\n{PAPER_ABSTRACT}\nSummarize this abstract in one sentence."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_tokens": 128,
        "session_id": "test_session_456"
    }
    
    t0 = time.time()
    try:
        ans_1 = post_request(API_URL, payload)
        print("--- Turn 1 Response ---")
        content_1 = ans_1["choices"][0]["message"]["content"]
        print(content_1)
        print(f"Time taken: {time.time() - t0:.2f}s\n")
    except Exception as e:
        print(f"Error on Turn 1: {e}")
        return
    
    # 2. Turn 2: Send follow-up request (reusing session_id)
    print("Sending Turn 2 (Follow-up question) with non-zero temp...")
    payload_2 = {
        "model": "diffkv-serving",
        "messages": [
            {"role": "user", "content": f"Here is the paper abstract:\n{PAPER_ABSTRACT}\nSummarize this abstract in one sentence."},
            {"role": "assistant", "content": content_1},
            {"role": "user", "content": "What is the main focus of this paper?"}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_tokens": 128,
        "session_id": "test_session_456"
    }
    
    t0 = time.time()
    try:
        ans_2 = post_request(API_URL, payload_2)
        print("--- Turn 2 Response ---")
        content_2 = ans_2["choices"][0]["message"]["content"]
        print(content_2)
        print(f"Time taken: {time.time() - t0:.2f}s\n")
    except Exception as e:
        print(f"Error on Turn 2: {e}")

if __name__ == "__main__":
    test_flow()
