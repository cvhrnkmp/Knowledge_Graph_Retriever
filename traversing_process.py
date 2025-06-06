import ollama
import numpy as np

import time
from sentence_transformers import util
from tqdm import tqdm
import json
import os
from utils.drift_search_prompt import DRIFT_PRIMER_PROMPT, DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT
from utils.MongoDB_Driver import MongoCRUD

from retriever.BM25_retriever import BM25_Retriever
from retriever.Connecting_Dots import ConnectingDotsRAG
from retriever.vector_search import VectorSearch 
from retriever.HyDE_retriever import HydeRetriever
from utils.rag_utils import RAGUtils

tqdm.pandas()
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from pymongo import MongoClient
import asyncio

class verdict(BaseModel):
    intermediate_answer:str
    score:int
    follow_up_queries: list[str]
    
class reduce(BaseModel):
    response:str    

class HydeRetriever():
    
    def __init__(self, llm, embd_model, k = 20):
        self.llm = llm
        self.embd_model = embd_model
        #self.vector_retriever = vector_retriever
        self.prompt_template = """Please write a passage to answer the question.
Question: {QUESTION}
Passage:"""
    def invoke(self, query):
        prompt = self.prompt_template.format(QUESTION=query)
        response = self.llm.get_response(prompt)
        #response = response[0]["generated_text"]#[-1]["content"]
        #print(response)
        #return self.vector_retriever.get_relevant_documents(response, ragflow)
        return response
    def load_and_init(self):
        pass
    
class MultiRetrieverEngine():

    def __init__(self, retriever, top_k=6):
        self.retriever = retriever
        self.top_k = top_k
    def get_relevant_documents(self, query):
        
        self.scored_docs = []
        
        for retriever in self.retriever:
            if isinstance(retriever, HydeRetriever):
                hyde_retriver = retriever.pop(retriever)
                hyde_query = hyde_retriver.invoke(query)
                query = hyde_query
        
        for retriever in self.retriever:
            retrieved_docs = retriever.get_relevant_documents(query=query)
            self.scored_docs.append(retrieved_docs)
        
        if len(self.retriever) > 1:
            self.normalized_docs = [RAGUtils.normalize_scores(relevant_docs) for relevant_docs in self.scored_docs]
            self.scored_docs = RAGUtils.score_fusion(self.normalized_docs)
            relevant_docs = self.scored_docs
            if len(relevant_docs) >= self.top_k:
                relevant_docs = relevant_docs[:self.top_k]
                
        elif len(self.retriever) == 1:
            retr = self.retriever[-1]
            
            # TODO:Add supporting chunks as list
            supporting_docs, retrieved_docs = retr.get_relevant_documents(query=query, raglfow=self)
            if len(retrieved_docs) >= self.top_k:
                relevant_docs = retrieved_docs[:self.top_k]
            else:
                relevant_docs = retrieved_docs

        return relevant_docs
    
    def calculate_score(self, query, dataframe, key=None):
        
        if isinstance(dataframe, list):
            if isinstance(dataframe[0], str):
                self.scores = []
                summed_scores = None
                for idx, retriever in enumerate(self.retriever):
                    if isinstance(retriever, HydeRetriever):
                        hyde_retriver = self.retriever.pop(idx)
                        hyde_query = hyde_retriver.invoke(query)
                        query = hyde_query
                
                for retriever in self.retriever:
                    scores = None
                    if isinstance(retriever, VectorSearch):
                        scores =  [retriever.calculate_score(query, x) for x in dataframe]
                    if isinstance(retriever, BM25_Retriever):
                        scores = retriever.calculate_score(query)
                        scores = [score[0] for score in scores]
                    self.scores.append(scores)
                    
                if len(self.retriever) > 1:
                    self.normalized_scores = [RAGUtils.normalize_scores(score) for score in self.scores]
                    #self.scored_docs = RAGUtils.score_fusion(self.normalized_docs)
                    #relevant_docs = self.scored_docs
                    summed_scores = [sum(x) for x in zip(*self.normalized_scores)]
                
                return summed_scores

        else:
        
            self.scores = []
            summed_scores = None
            for idx, retriever in enumerate(self.retriever):
                if isinstance(retriever, HydeRetriever):
                    hyde_retriver = self.retriever.pop(idx)
                    hyde_query = hyde_retriver.invoke(query)
                    query = hyde_query
            
            for retriever in self.retriever:
                scores = []
                if isinstance(retriever, VectorSearch):
                    scores = dataframe[key].apply(lambda x: retriever.calculate_score(query, x)).tolist()
                if isinstance(retriever, BM25_Retriever):
                    scores = retriever.calculate_score(query)
                    scores = [score[0] for score in scores]
                self.scores.append(scores)
                
            if len(self.retriever) > 1:
                self.normalized_scores = [RAGUtils.normalize_scores(score) for score in self.scores]
                #self.scored_docs = RAGUtils.score_fusion(self.normalized_docs)
                #relevant_docs = self.scored_docs
                summed_scores = [sum(x) for x in zip(*self.normalized_scores)]
            
            return summed_scores


class DRIFTSearch():
    
    def __init__(self, chat_model:str, embeddings_model:str, dataset:str):
        
        self.chat_model_name = chat_model
        self.embeddings_model_name= embeddings_model
        self.embeddings_model = OllamaEmbedding(model = embeddings_model)
        self.chat_model = OllamaChat(model = chat_model)
        self.dataset = dataset
        
        entities, relation, communities = self.get_data(chat_model, self.dataset)
        
        self.entities = entities
        self.relation = relation
        self.communities = communities
        #self.communities["community"] = self.communities["community"].apply(lambda x: x.split(";"))
        self.documents = self.get_documents()
        
    def get_documents(self):
        self.mongo_crud = MongoCRUD(
            username=os.getenv("MONGODB_USERNAME"),
            password=os.getenv("MONGODB_PASSWORD"),
            cluster_url=os.getenv("MONGODB_CLUSTER"),
            database_name=os.getenv("MONGODB_DATABASE", "From_Local_to_Global"),
            default_collection_name="Documents"
        )
        return self.mongo_crud.get_all_documents(filter = {}, collection_name="Documents", as_lg_document=True)
    
    def set_test_series(self, test:str, connection_settings:Dict):
        self.test_series = test
        self.mongo_crud = MongoCRUD(
            username=os.getenv("MONGODB_USERNAME"),
            password=os.getenv("MONGODB_PASSWORD"),
            cluster_url=os.getenv("MONGODB_CLUSTER"),
            database_name="Connecting_the_Dots",
            default_collection_name="Ragflow_Log"
        )
    
    def get_data(self, model:str, dataset:str):
        
        username = "user"
        password = "pass"
        cluster_url = "localhost"
        database = "From_Local_to_Global"
        
        connection_string = f"mongodb://{username}:{password}@{cluster_url}:27019/?retryWrites=true&w=majority&directConnection=true"
        client = MongoClient(connection_string)
        # Select the database and collection
        db = client[database]
        col_entities = db["Entities"]
        col_relation = db["Relations"]
        col_communities = db["Communities"]

        # Define your filter - adjust this as per your needs
        #filter_query = {"$text": {"$search": "data"}}  # Replace 'field_name' with the actual field

        # Execute the query
        documents_entities = list(col_entities.find({}))
        documents_relation = list(col_relation.find({}))
        documents_communities = list(col_communities.find({}))

        # Convert to DataFrame
        df_entities = pd.DataFrame(documents_entities)
        df_relation = pd.DataFrame(documents_relation)
        df_communities = pd.DataFrame(documents_communities)
        
        df_entities = df_entities[df_entities["dataset"] == dataset]
        df_relation = df_relation[df_relation["dataset"] == dataset]
        #df_communities = df_communities[(df_communities["model"] == model) & (df_communities["dataset"] == dataset)]
        # df_communities = df_communities[(df_communities["model"] == model) & (df_communities["dataset"] == dataset)]
        df_communities = df_communities[df_communities["dataset"] == dataset]
        
        df_entities["description"]=df_entities.apply(lambda row: row["description"].strip("] ["), axis=1)
        df_relation["relation_description"]=df_relation.apply(lambda row: row["relation_description"].strip("] ["), axis=1)
        df_communities["community_desc"]=df_communities.apply(lambda row: row["community_desc"].strip("] ["), axis=1)
        
        return df_entities, df_relation, df_communities
    
    def log_step(self, log:Dict):
        
        for key, value in log.items(): 
            self.log[key] = value
    
    def write_log(self):
        if self.log is not None:
            self.mongo_crud.insert(
                data=self.log,
                collection_name="Ragflow_Log"
                
            )
    def init_search(self, community_search, entity_search, relation_search, document_search, **kwargs):
        #relation_llama
        self.entities["embd_desc"] = self.entities["description"].progress_apply(lambda x: self.embeddings_model.get_embd(x))
        self.relation["embd_desc"] = self.relation["relation_description"].progress_apply(lambda x: self.embeddings_model.get_embd(x))
        
        embd_model = self.embeddings_model #kwargs.get("embedding_model")
        llm = self.chat_model #kwargs.get("chat_model")
        
        for doc in tqdm(self.documents):
            doc.metadata["text_embd"] = self.embeddings_model.get_embd(doc.page_content)
        
        ## -- Community Level -- ## 
        if community_search == "vector":
            self.community_search = VectorSearch(embd_model, k=20)
        elif community_search == "hybrid":
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.communities["community_desc"].tolist())
            self.community_search = MultiRetrieverEngine(retriever=[vector_retr, bm25])
        
        ## -- Entitiy Level -- ## 
        if entity_search == "vector":
            self.entity_search = VectorSearch(embd_model, k=20)
        elif entity_search == "hybrid":
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.entities["description"].tolist())
            self.entity_search = MultiRetrieverEngine(retriever=[vector_retr, bm25])
        elif entity_search == "hybrid_hyde":
            hyde = HydeRetriever(llm, embd_model)    
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.entities["description"].tolist())
            self.entity_search = MultiRetrieverEngine(retriever=[hyde, vector_retr, bm25])
            
        ## -- Relation Level -- ## 
        if relation_search == "vector":
            self.relation_search = VectorSearch(embd_model, k=20)
        elif relation_search == "hybrid":
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.relation["relation_description"].tolist())
            self.relation_search = MultiRetrieverEngine(retriever=[vector_retr, bm25])
        elif relation_search == "hybrid_hyde":
            hyde = HydeRetriever(llm, embd_model)    
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.relation["relation_description"].tolist())
            self.relation_search = MultiRetrieverEngine(retriever=[hyde, vector_retr, bm25])    
        
        ## -- Document Level -- ## 
        if document_search == "vector":
            vector_retriever = VectorSearch(embd_model, k=20)    
            self.document_search = vector_retriever
        elif document_search == "hybrid":
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.documents)
            self.document_search = MultiRetrieverEngine(retriever=[vector_retr, bm25])
        elif document_search == "hybrid_hyde":
            hyde = HydeRetriever(llm, embd_model)    
            vector_retr = VectorSearch(embd_model, k=20)
            bm25 = BM25_Retriever(k=20)
            bm25.load_and_init(self.documents)
            self.document_search = MultiRetrieverEngine(retriever=[hyde, vector_retr, bm25])
        elif document_search == "connecting_dots":    
            connecting_dot_retriever = ConnectingDotsRAG(
                embd_model=embd_model,
                hidden_size=768
            )
            connecting_dot_retriever.load_and_init()
            self.document_search = connecting_dot_retriever
        
    
    async def search(self, global_query: str) -> dict:
        import asyncio
        start =  time.perf_counter()
        # Test-series logging
        if hasattr(self, "test_series"):
            self.log = {
                "name": self.test_series,
                "query": global_query,
                "model": self.chat_model_name,
                "embeddings_model": self.embeddings_model_name,
            }

        # Expand and decompose the query
        hype_answer, df_top_k = await asyncio.to_thread(self.expand_query, global_query, 5)
        expanded_time = time.perf_counter() - start
        results = await asyncio.to_thread(self.decompose_query, global_query, df_top_k)
        decompose_time = time.perf_counter() - start - expanded_time
        follow_up_queries = results["follow_up_queries"]
        if hasattr(self, "log"):
            self.log_step({"community_response": results["intermediate_answer"]})
            self.log_step({"follow_up_queries": follow_up_queries})

        # Compute relevant entities
        community_array = df_top_k["community"].apply(lambda x: [e.strip("'' ") for e in x])
        relevant_entities = [e for com in community_array for e in com]
        print("Expanded time", expanded_time)
        print("Decompose time", decompose_time)

        async def _handle_query(q: str) -> dict:
            # Entity scoring
            
            ents = self.entities[self.entities["entity"].isin(relevant_entities)].copy()
            escores = await asyncio.to_thread(self.entity_search.calculate_score, q, ents, "embd_desc")
            ents["score"] = pd.to_numeric(escores)
            top_ents = ents.nlargest(5, "score")

            # Relation & target scoring
            rels = self.relation[self.relation["anchor_entity"].isin(top_ents["entity"].values)].copy()
            targs = self.entities[self.entities["entity"].isin(rels["target_entity"].values)].copy()
            rel_task = asyncio.to_thread(self.relation_search.calculate_score, q, rels, "embd_desc")
            targ_task = asyncio.to_thread(self.entity_search.calculate_score, q, targs, "embd_desc")
            if isinstance(self.document_search, ConnectingDotsRAG):
                doc_task = asyncio.to_thread(self.document_search.get_relevant_documents, q)
            else:
                texts = [d.page_content for d in self.documents]
                doc_task = asyncio.to_thread(self.document_search.calculate_score, q, texts)

            rel_scores, targ_scores, doc_res = await asyncio.gather(rel_task, targ_task, doc_task)
            rels["score"] = pd.to_numeric(rel_scores)
            top_rels = rels.nlargest(5, "score")
            targs["score"] = pd.to_numeric(targ_scores)
            top_targs = targs.nlargest(5, "score")

            # Document selection
            if isinstance(self.document_search, ConnectingDotsRAG):
                supp, scored = doc_res
                docs = [doc for _, doc in scored]
                docs_df = pd.DataFrame([{"text": d.page_content, "score": 0} for d in docs + supp])
            else:
                dscores = doc_res
                docs_df = pd.DataFrame([{"text": texts[i], "score": dscores[i]} for i in range(len(texts))])
            top_docs = docs_df.nlargest(5, "score")

            # Generate local response
            return await asyncio.to_thread(
                self.map_local_response,
                query=global_query,
                anchor_descriptions=top_ents["description"].tolist(),
                relation_descriptions=top_rels["relation_description"].tolist(),
                target_descriptions=top_targs["description"].tolist(),
                documents=top_docs["text"].tolist(),
            )

        # Execute tasks concurrently
        responses = await asyncio.gather(*[asyncio.create_task(_handle_query(q)) for q in follow_up_queries])
        handle_time = time.perf_counter() - start- decompose_time
        if hasattr(self, "log"):
            self.log_step({"intermediadte_response": [r["intermediate_answer"] for r in responses]})

        # Global reduction
        global_resp = await asyncio.to_thread(
            self.reduce_global_response,
            community_1_answer=results["intermediate_answer"],
            entity_level_answer=[r["intermediate_answer"] for r in responses],
        )
        global_time = time.perf_counter() - start - handle_time
        if hasattr(self, "log"):
            self.log_step({"global_response": global_resp})
            self.write_log()
        print("Handle time", handle_time)    
        print("Global time", global_time)
        
        return global_resp

    def calculate_sim_score(self, query_vector, document_vector): 
        return float(util.cos_sim(query_vector, document_vector)[0][0])

    def expand_query(self, query: str, k:int=5) -> tuple[str, dict[str, int]]:
            """
            Expand the query using a random community report template.

            Args:
                query (str): The original search query.

            Returns
            -------
            tuple[str, dict[str, int]]: Expanded query text and the number of tokens used.
            """

            prompt = f"""Create a hypothetical answer to the following query: {query}\n\n
                                        Ensure that the hypothetical answer does not reference new named entities that are not present in the original query."""
                                        
            
            model_response = self.chat_model.get_response(prompt)
            query_embd = self.embeddings_model.get_embd(query)
            response_embd = self.embeddings_model.get_embd(model_response)
            
            # compute similarity scores for each community
            if isinstance(self.community_search, VectorSearch):
                # vector search: score each community description string
                scores = [self.community_search.calculate_score(model_response, desc)
                          for desc in self.communities["community_desc"].tolist()]
            else:
                # multi retriever: use DataFrame and key
                scores = self.community_search.calculate_score(model_response, self.communities, "community_desc")
            self.communities["score"] = scores
            
            top_k_coms = self.communities.nlargest(k, "score")
            return model_response, top_k_coms

    def decompose_query( self,
            query: str, reports: pd.DataFrame
        ) -> tuple[dict, dict[str, int]]:
            """
            Decompose the query into subqueries based on the fetched global structures.

            Args:
                query (str): The original search query.
                reports (pd.DataFrame): DataFrame containing community reports.

            Returns
            -------
            tuple[dict, int, int]: Parsed response and the number of prompt and output tokens used.
            """
            community_reports = "\n\n".join(reports["community_desc"].tolist())
            prompt = DRIFT_PRIMER_PROMPT.format(
                query=query, community_reports=community_reports
            )

            model_response = self.chat_model.get_response(prompt, format=verdict)
            #print(model_response)
            #response = model_response.output.content

            parsed_response = json.loads(model_response)
            #print(parsed_response)

            return parsed_response
        
    def map_local_response(
            self,
            query: str,
            anchor_descriptions: List[str],
            relation_descriptions: List[str],
            target_descriptions: List[str],
            documents: List[str]
        ) -> tuple[dict, dict[str, int]] :
        anchor_desc = "\n\n".join(anchor_descriptions)
        relation_desc = "\n\n".join(relation_descriptions)
        target_desc = "\n\n".join(target_descriptions)
        docs = "\n\n".join(documents)
        
        context_data = anchor_desc + "\n\n" + relation_desc + "\n\n" +  target_desc + "\n\n" + docs
        
        prompt = DRIFT_LOCAL_SYSTEM_PROMPT.format(
            global_query=query, context_data=context_data, response_type = verdict.model_json_schema()
        )

        model_response = self.chat_model.get_response(prompt, format=verdict)
        #print(model_response)
        #response = model_response.output.content

        parsed_response = json.loads(model_response)
        #print(parsed_response)

        return parsed_response

    def reduce_global_response(
            self,
            community_1_answer:str,
            entity_level_answer:List[str],
        ) -> tuple[dict, dict[str, int]] :
            
        context_data = community_1_answer + "\n\n" + "\n\n".join(entity_level_answer)
        
        prompt = DRIFT_REDUCE_PROMPT.format(
            context_data=context_data, response_type = reduce.model_json_schema()
        )

        model_response = self.chat_model.get_response(prompt, format=reduce)
        #print(model_response)
        #response = model_response.output.content

        parsed_response = json.loads(model_response)
        #print(parsed_response)

        return parsed_response


class OllamaChat:
    def __init__(self, model: str = 'llama3.2'):
        """
        Initialize the OllamaChat with the specified model.

        Args:
            model (str): The name of the model to use. Default is 'llama3.2'.
        """
        self.model = model

    def get_response(self, prompt: str, format = None) -> str:
        """
        Generate a response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response from the model.
        """
        if format == None:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            #return response["message"]["content"]
            return response.message.content
        else:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}], format=format.model_json_schema())
            #return response["message"]["content"]
            return response.message.content
        
      
class OllamaEmbedding:
    def __init__(self, model):
        self.model = model
    
    def get_embd(self, input):
        ollama_embeddings = ollama.embeddings(model=self.model, prompt=input)
        return np.array(ollama_embeddings['embedding'])#.reshape(1, -1)
    
    def embed_text(self, input):
        return self.get_embd(input)
    
    def embed_list(self, input:List[str]):
        return [self.get_embd(x) for x in input]