from typing import List, Tuple
from langchain.docstore.document import Document
from .Base_Retriever import BaseRetriever
from langchain.retrievers import BM25Retriever
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BM25_Retriever(BaseRetriever):
    
    def __init__(self, k=20):
        self.k = k
        self.scaler = MinMaxScaler()
        
    def load_and_init(self, ragflow):
        if isinstance(ragflow, list):
            if isinstance(ragflow[0], str):
                self.BM25 = BM25Retriever.from_texts(ragflow)
            else:
                self.BM25 = BM25Retriever.from_documents(ragflow)
        else:
            self.BM25 = BM25Retriever.from_documents(ragflow.doc_chunks)
    
    def get_relevant_documents(self, query:str, *args, **kwargs) -> List[Tuple[float, Document]]:
        scores = self.BM25.vectorizer.get_scores(query.split(" "))
        #print("scores_un: ", scores)
        scores = self.normalize_score(scores.reshape(-1, 1))
        #print("scores: ", scores)
        top_n = np.argsort(scores.ravel())#[::-1]
        #print("top_n: ", top_n)
        return [(scores[i], self.BM25.docs[i]) for i in top_n][:self.k]
        #return [(score, document) for idx, docu
    
    def normalize_score(self, scores):
        return self.scaler.fit_transform(scores)
    
    def calculate_score(self, query:str):
        scores = self.BM25.vectorizer.get_scores(query.split(" "))
        #print("scores_un: ", scores)
        scores = self.normalize_score(scores.reshape(-1, 1))
        
        return scores