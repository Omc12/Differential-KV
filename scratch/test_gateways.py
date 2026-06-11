import requests
import json

def query_server(port, prompt, stream=False):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 100
    }
    
    try:
        if stream:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            text = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            text += content
                            print(content, end="", flush=True)
                        except Exception as e:
                            pass
            print()
            return text
        else:
            response = requests.post(url, headers=headers, json=payload)
            res_json = response.json()
            if "choices" in res_json:
                return res_json["choices"][0]["message"]["content"]
            else:
                return str(res_json)
    except Exception as e:
        return f"ERROR: {e}"

prompt = "Hello, translate 'hello' to French."
print(f"--- Querying Python Server on 9100 with '{prompt}' ---")
py_res = query_server(9100, prompt)
print(f"Python Response:\n{py_res}\n")

print(f"--- Querying C++ Server on 9099 with '{prompt}' ---")
cpp_res = query_server(9099, prompt)
print(f"C++ Response:\n{cpp_res}\n")
