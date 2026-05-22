# LangChain Agent Example
from integrations.langchain_sparse_connector import DiffKVSparseLLM
llm = DiffKVSparseLLM()
print(llm.invoke("What is the capital of France?"))
