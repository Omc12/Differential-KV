import urllib.request
import json

def query_server(port, prompt):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 100
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            if "choices" in res_json:
                return res_json["choices"][0]["message"]["content"]
            else:
                return res_body
    except Exception as e:
        return f"ERROR: {e}"

prompt = "Hello, translate 'hello' to French."
print(f"--- Querying Python Server on 9100 with '{prompt}' ---")
py_res = query_server(9100, prompt)
print(f"Python Response:\n{py_res}\n")

print(f"--- Querying C++ Server on 9099 with '{prompt}' ---")
cpp_res = query_server(9099, prompt)
print(f"C++ Response:\n{cpp_res}\n")
