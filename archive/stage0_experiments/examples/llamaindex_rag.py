# LlamaIndex RAG Example
from integrations.llamaindex_query_adapter import DiffKVLlamaIndexAdapter
llm = DiffKVLlamaIndexAdapter()
print(llm.complete("Summarize the Differential KV paper."))
