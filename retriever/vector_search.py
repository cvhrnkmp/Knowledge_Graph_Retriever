from typing import List, Any
from langchain.docstore.document import Document
from .Base_Retriever import BaseRetriever
from sentence_transformers import util
from tqdm import tqdm
from functools import singledispatchmethod
import pandas as pd

class VectorSearch(BaseRetriever):
    def __init__(self, model, k=20):
        self.model = model
        self.k= k
        #self.similarity_fn = SimilarityFunction.DOT_PRODUCT
        
    def calculate_sim_score(self, query_vector, document_vector): 
        return float(util.cos_sim(query_vector, document_vector)[0][0])
    
    
    # def calculate_score(self, query, phrase):
    #     if isinstance(phrase, str):
    #         return self.calculate_sim_score(self.model.embed_text(query), self.model.embed_text(phrase))
    #     else:
    #         return self.calculate_sim_score(self.model.embed_text(query), phrase)
    
    # def calculate_score(self, query:str, dataframe:pd.DataFrame, key:str):
    #     scores = [self.calculate_score(query, desc)
    #                       for desc in dataframe[key].tolist()]
    #     return scores
    
    def calculate_score(self, query: str, items: Any, key: str = None):
        """Compute similarity score(s) for a single text or vector, a list, or a DataFrame column."""
        # Handle DataFrame column
        if isinstance(items, pd.DataFrame):
            values = items[key].tolist()
            return [self.calculate_score(query, v) for v in values]
        # Handle list of texts or embeddings
        if isinstance(items, list):
            return [self.calculate_score(query, v) for v in items]
        # Single string or precomputed embedding
        query_vec = self.model.embed_text(query)
        if isinstance(items, str):
            item_vec = self.model.embed_text(items)
        else:
            item_vec = items
        return self.calculate_sim_score(query_vec, item_vec)
    
    def get_relevant_documents(self, query, ragflow) -> list(tuple()): 
        query_vector = ragflow.embeddings_model.embed_text(query)
        documents = ragflow.doc_chunks.copy()
        scored_docs = [(self.calculate_sim_score(query_vector, doc.metadata["embedding"]), doc) for doc in tqdm(documents)]
        sorted_scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        return sorted_scored_docs[:self.k]