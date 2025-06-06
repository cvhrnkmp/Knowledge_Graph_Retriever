from .Base_Retriever import BaseRetriever
from langchain.chains import HypotheticalDocumentEmbedder

class HydeRetriever(BaseRetriever):
    
    def __init__(self, llm, embd_model, vector_retriever):
        self.llm = llm
        self.embd_model = embd_model
        self.vector_retriever = vector_retriever
        self.prompt_template = """Please write a passage to answer the question.
Question: {QUESTION}
Passage:"""
    def get_relevant_documents(self, query):
        prompt = self.prompt_template.format(QUESTION=query)
        response = self.llm.invoke(prompt)[0]["generated_text"][-1]["content"]
        print(response)
        return self.vector_retriever.get_relevant_documents(response)
    def load_and_init(self, ragflow):
        pass
    