import os
from typing import List, Dict
from integrations.langchain_adapter import DiffKVLangChainAdapter

class RepositoryMemoryAPI:
    """
    High-level API for using DiffKV as a persistent memory for repository-scale coding.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.adapter = DiffKVLangChainAdapter()
        self.session_id = None

    def initialize_session(self):
        import requests
        response = requests.post("http://localhost:8000/v1/sessions")
        self.session_id = response.json()["session_id"]
        return self.session_id

    def ingest_file(self, file_path: str):
        full_path = os.path.join(self.repo_path, file_path)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Ingesting a file into DiffKV's sparse memory
        # We do this by sending the content as a "system" context or similar
        prompt = f"INGEST FILE: {file_path}\nCONTENT:\n{content}\n"
        return self.adapter.predict(prompt, max_tokens=10)

    def ask_about_repo(self, question: str) -> str:
        return self.adapter.predict(f"REPOSITY CONTEXT ACTIVE. Question: {question}")

if __name__ == "__main__":
    # api = RepositoryMemoryAPI("./")
    # api.initialize_session()
    # api.ingest_file("serving/real_sparse_serving_runtime.py")
    # print(api.ask_about_repo("What does the RealSparseServingRuntime do?"))
    pass
