# OpenAI SDK Client Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-unused")
response = client.chat.completions.create(
    model="dkv",
    messages=[{"role": "user", "content": "How does sparse KV work?"}]
)
print(response.choices[0].message.content)
