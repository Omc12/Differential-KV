import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload = {
    "model": "Qwen2.5-0.5B-Instruct",
    "messages": [
        {"role": "user", "content": "Hello, what is your name?"}
    ],
    "max_tokens": 10,
    "stream": False
}

response = requests.post(url, headers=headers, json=payload)
print("Status Code:", response.status_code)
print("Response JSON:")
print(json.dumps(response.json(), indent=2))
