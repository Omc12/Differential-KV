# LangChain Agent Example
from integrations.langchain_sparse_connector import DKVSparseLLM
llm = DKVSparseLLM()
print(llm.invoke("What is the capital of France?"))
