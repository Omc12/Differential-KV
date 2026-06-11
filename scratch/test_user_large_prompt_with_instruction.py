import urllib.request
import json
import time

url = "http://localhost:8000/v1/chat/completions"

paper_abstract = """Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate an. """

prompt = paper_abstract * 34 + "\n\nBased on the text above, answer this question: What is the main technique proposed in the paper to accelerate training?"

payload = {
    "model": "diffkv-native-qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
    "max_tokens": 128
}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Sending request with large prompt + instruction...")
start = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as response:
        dur = time.time() - start
        res_body = response.read().decode('utf-8')
        res_json = json.loads(res_body)
        print(f"Success in {dur:.2f}s")
        print("Response:")
        print(res_json["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Request failed: {e}")
