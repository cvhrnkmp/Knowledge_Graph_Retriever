from typing import List,Tuple
from langchain.docstore.document import Document
from prompt_templates.rag_template import SYSTEM_RAG_PROMPT
from langchain.load import dumps, loads
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RAGUtils():
    
    def __init__(self):
        pass
        
    def score_fusion(results):
        fused_scores = {}
        for docs in results:
            # Assumes the docs are returned in sorted order of relevance
            #print(docs)
            for score, doc in docs:
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += score
        
        reranked_results = [
            (score, loads(doc))
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results
    
    def reciprocal_rank_fusion(results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            # Assumes the docs are returned in sorted order of relevance
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (score, loads(doc))
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results
    
    def lost_in_the_middle(documents:List[tuple]):
        
        indeces = sorted([3, 5, 6, 8], reverse=True)
        
        reorderd_set = documents.copy()
        #print(type(reorderd_set))
        #print(reorderd_set[0])
        for idx in indeces:
            #print(str(len(reorderd_set)))
            if len(reorderd_set) > idx:
                #print(str(len(reorderd_set)))
                entry = reorderd_set.pop(idx)
                reorderd_set.append(entry)       
        
        return reorderd_set
    
    def prepare_rag_prompt(query, context, supporting_docs = None):
        complete_query = None
        if supporting_docs:
            complete_query = "Kontext: " + "\n".join([doc.page_content for score, doc in context]) 
            complete_query = "Unterstützender Kontext: " + "\n".join([doc.page_content for _, doc in supporting_docs]) 
            complete_query = complete_query+ "\n" + "Anfrage: " + query
        else:        
            complete_query = "Kontext: " + "\n".join([doc.page_content for score, doc in context]) + "\n" + "Anfrage: " + query 
        #return reordered_docs
        messages = [
            {"role": "system", "content": SYSTEM_RAG_PROMPT},
            #{"role": "user", "content": " )},
            {"role": "user", "content": complete_query}
        ]
        return messages
    
    def normalize_scores(relevant_documents:List[Tuple[float, Document]]):
        
        if isinstance(relevant_documents, list) and len(relevant_documents) > 0:
            if isinstance(relevant_documents[-1], tuple):
                scores = np.array([score for score,_ in relevant_documents]).reshape(-1,1)
                
                scaler = MinMaxScaler()
                scaled_scores = scaler.fit_transform(scores)
                
                return [(float(n_score), doc) for (n_score, (score, doc)) in zip(scaled_scores, relevant_documents)]
            else:
                scores = np.array([score for score in relevant_documents]).reshape(-1,1)
                
                scaler = MinMaxScaler()
                scaled_scores = scaler.fit_transform(scores)
                
                return [float(n_score) for (n_score, score) in zip(scaled_scores, relevant_documents)]