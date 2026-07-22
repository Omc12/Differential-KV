# LlamaIndex RAG Example
from integrations.llamaindex_query_adapter import DKVLlamaIndexAdapter
llm = DKVLlamaIndexAdapter()
print(llm.complete("Summarize the Differential KV paper."))
