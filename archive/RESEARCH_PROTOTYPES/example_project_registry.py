"""
example_project_registry.py

Registry of runnable examples for Differential KV.
Provides real usage references for major integrations.
"""

import os
from typing import List, Dict, Any

class ExampleProjectRegistry:
    """
    Curates and manages example scripts.
    """
    def __init__(self, examples_dir: str = "examples"):
        self.examples_dir = examples_dir
        os.makedirs(examples_dir, exist_ok=True)

    def register_default_examples(self):
        """Creates the initial set of integration examples."""
        examples = {
            "hf_integration.py": self._get_hf_example(),
            "openai_sdk_client.py": self._get_openai_example(),
            "langchain_agent.py": self._get_langchain_example(),
            "llamaindex_rag.py": self._get_llamaindex_example()
        }
        
        for name, content in examples.items():
            path = os.path.join(self.examples_dir, name)
            with open(path, "w") as f:
                f.write(content)

    def _get_hf_example(self) -> str:
        return """# HuggingFace Integration Example
from integrations.huggingface_runtime_adapter import DiffKVHFAdapter
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"
model = DiffKVHFAdapter.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
"""

    def _get_openai_example(self) -> str:
        return """# OpenAI SDK Client Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-unused")
response = client.chat.completions.create(
    model="diff-kv",
    messages=[{"role": "user", "content": "How does sparse KV work?"}]
)
print(response.choices[0].message.content)
"""

    def _get_langchain_example(self) -> str:
        return """# LangChain Agent Example
from integrations.langchain_sparse_connector import DiffKVSparseLLM
llm = DiffKVSparseLLM()
print(llm.invoke("What is the capital of France?"))
"""

    def _get_llamaindex_example(self) -> str:
        return """# LlamaIndex RAG Example
from integrations.llamaindex_query_adapter import DiffKVLlamaIndexAdapter
llm = DiffKVLlamaIndexAdapter()
print(llm.complete("Summarize the Differential KV paper."))
"""

if __name__ == "__main__":
    registry = ExampleProjectRegistry()
    registry.register_default_examples()
    print("Examples registered.")
