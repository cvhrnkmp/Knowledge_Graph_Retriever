from retriever.utils.GraphRAGBaseModel import GraphRAGBaseModel
from retriever.vector_search import VectorSearch
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from uuid import uuid4
import networkx as nx
import nltk
import spacy
from HanTa import HanoverTagger as ht
from retriever.utils.GraphSAGE import GraphSAGEEmbeddings
from retriever.Base_Retriever import BaseRetriever
from utils.pre_processing_utils import TextRank
from langchain_community.retrievers import BM25Retriever
from nltk.stem import WordNetLemmatizer
from sentence_transformers import util
from tqdm import tqdm
from utils.MongoDB_Driver import MongoCRUD
from utils.neo4j_import import Neo4jGraphInserter
from neo4j import GraphDatabase
from utils.pre_processing_utils import clean_and_normalize

class CustomBM25(BM25Okapi):
    
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)
    
    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

class ConnectingDotsRAG(GraphRAGBaseModel, BaseRetriever):
    
    def __init__(self, embd_model, hidden_size):
        self.embd_model = embd_model
        self.hidden_size = hidden_size
        self.vector_store_nodes = None
        self.GraphSAGE = GraphSAGEEmbeddings(self.embd_model)
        self.TextRank = TextRank(self.embd_model)
        self.hanta_ger = ht.HanoverTagger('morphmodel_ger.pgz')
        self.lemmatizer= WordNetLemmatizer()
        
        
        # Initialize the CRUD handler with a default collection.
        # self.mongo_crud = MongoCRUD(USERNAME, PASSWORD, CONNECTION_STRING, DATABASE_NAME, DEFAULT_COLLECTION_NAME)
        # self.neo4j_inserter = Neo4jGraphInserter("bolt://localhost:7688", "neo4j", "password")
        #self.G=self.neo4j_to_networkx()  
    def load_and_init(self, ragflow=None):
        # Replace these with your actual MongoDB Atlas connection details.
        USERNAME = "user"
        PASSWORD = "pass"
        CONNECTION_STRING = "localhost"
        DATABASE_NAME = "Connecting_the_Dots"
        DEFAULT_COLLECTION_NAME = "Text_Chunks"

        self.mongo_crud = MongoCRUD(USERNAME, PASSWORD, CONNECTION_STRING, DATABASE_NAME, DEFAULT_COLLECTION_NAME)
        self.neo4j_inserter = Neo4jGraphInserter("bolt://localhost:7689", "neo4j", "password")
        self.G = self.neo4j_to_networkx()
        #self.construct_kg(ragflow.doc_chunks)
        
    def neo4j_to_networkx(self, url="bolt://localhost:7689", user="neo4j", password="password"):
        """
        Connects to a Neo4j database, fetches the nodes and relationships,
        and converts them into a NetworkX graph.

        :param uri: Neo4j URI (e.g., "bolt://localhost:7687")
        :param user: Neo4j username
        :param password: Neo4j password
        :return: A NetworkX graph
        """
        
        # Create a directed graph (change to nx.Graph() if undirected is needed)
        G = nx.Graph()  

        # Connect to Neo4j
        driver = GraphDatabase.driver(url, auth=(user, password))

        with driver.session() as session:
            # Fetch all nodes
            query_nodes = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"
            result_nodes = session.run(query_nodes)
            
            for record in tqdm(result_nodes, desc="Load Nodes from Graph Database"):
                labels = record["labels"][0]
                properties = record["properties"]
                node_id = properties["id"]
                G.add_node(node_id, type=labels, **properties)

            # Fetch all relationships
            query_edges = "MATCH (n)-[r]->(m) RETURN properties(n) AS source, properties(m) AS target, TYPE(r) AS type, properties(r) AS properties"
            result_edges = session.run(query_edges)

            for record in tqdm(result_edges, desc="Load Edges from Graph Databse"):
                source = record["source"]["id"]
                target = record["target"]["id"]
                rel_type = record["type"]
                properties = record["properties"]
                G.add_edge(source, target, type=rel_type, **properties)

        driver.close()
        return G
    
    def neo4j_networkx_reference(self, url="bolt://localhost:7688", user="neo4j", password="password"):
        """
        Connects to a Neo4j database, fetches the specified nodes and relationships,
        and converts them into a NetworkX graph.

        :param url: Neo4j URI (e.g., "bolt://localhost:7687")
        :param user: Neo4j username
        :param password: Neo4j password
        :return: A NetworkX graph
        """
        
        # Create a directed graph (change to nx.Graph() if undirected is needed)
        G = nx.Graph()

        # Connect to Neo4j
        driver = GraphDatabase.driver(url, auth=(user, password))

        with driver.session() as session:
            # Fetch nodes and relationships using the specified Cypher query
            query = """
            MATCH (n)-[r:REFERENCE_TO]-(m)  
            OPTIONAL MATCH (m)-[r2]-(o)  
            RETURN n, r, m, r2, o
            """
            result = session.run(query)
            
            for record in tqdm(result, desc="Loading Graph Data from Neo4j"):
                nodes = [record["n"], record["m"], record["o"]]
                relationships = [record["r"], record["r2"]]
                
                # Add nodes to the graph
                for node in nodes:
                    if node:  # Check if node is not None
                        node_id = node.id  # Get the internal Neo4j ID
                        properties = dict(node)  # Extract properties
                        labels = list(node.labels)[0] if node.labels else "Unknown"
                        G.add_node(node_id, type=labels, **properties)
                
                # Add relationships to the graph
                for rel in relationships:
                    if rel:  # Check if relationship is not None
                        source = rel.start_node.id
                        target = rel.end_node.id
                        rel_type = rel.type
                        properties = dict(rel)
                        G.add_edge(source, target, type=rel_type, **properties)
        
        driver.close()
        return G

        
    def get_relevant_documents(self, query, ragflow=None, **kwargs):
        ## -- Encode query
        embbed_query = np.asarray(self.embd_model.embed_text(query), dtype=np.float64)
        
        ## Extract Keywords
        keywords_query = self.TextRank.text_rank_with_bert_german(query, top_n=3)        
        keywords_query = list(set([self.lemmatizer.lemmatize(k) for k in keywords_query]))
        
        if ragflow:
            ragflow.log_step({"keywords_of_query": keywords_query})
        ## -- Retrieve top-k 
        #relevant_nodes = self.vector_store_nodes.similarity_search_with_score(query, k=5)
        
        # relevant_nodes = []
        # for node, data in self.node_list.items():
        #     score = util.cos_sim(embbed_query, data["graph_embd"])
        #     relevant_nodes.append((score[0], {node:self.node_list[node]}))
            
        #relevant_nodes = sorted(relevant_nodes, key=lambda x: x[0], reverse = True)[:5]
        
        ### S_s ###
        relevant_nodes = self.mongo_crud.cosine_similarity_search(
            embbed_query,
            top_k = 20,
            collection_name= "Nodes",
            index_name= "default",
            path = "graph_embd"
        )
        
        #ragflow.log_step({"relevant_nodes": relevant_nodes})
        
        #relevant_docs = self.vector_store_text.similarity_search_with_score(query, k=5)
        # relevant_docs = self.mongo_crud.cosine_similarity_search(
        #     embbed_query,
        #     top_k = 20,
        #     collection_name= "Nodes",
        #     index_name= "default",
        #     path = "embedding",
        #     filter = {"metadata.type":"text_chunk"}
        # )
        
        ## S_k ## 
        relevant_keywords = []
        for keyword in keywords_query:
            #retrieved_keywords = self.vector_store_keywords.similarity_search_with_score(keyword, k=2)
            embbed_keyword = np.asarray(self.embd_model.embed_text(keyword), dtype=np.float64)
            retrieved_keywords = self.mongo_crud.cosine_similarity_search(
                embbed_keyword,
                top_k = 1,
                collection_name= "Nodes",
                index_name= "default",
                path = "embedding",
                filter={"metadata.type": "keyword"}
            )
            relevant_keywords.extend(retrieved_keywords)
        
        relevant_neighbors = [] # R_d
        relevant_neighbors_referenced = []  # R_r
        relevant_neighbors_similar = []     # R_s
        ## -- Travers neighborhood of most-relevant nodes
        try:
            for idx, (score, re_node) in enumerate(relevant_nodes):
                #relevant_neighbors.append(self.G.nodes[re_node[0].metadata["uuid"]])
                if isinstance(re_node, list):
                    continue
                #neighborhood = self.G.neighbors(re_node[0].metadata["uuid"])
                node_uuid = re_node.metadata["uuid"]
                
                neighborhood = self.G.neighbors(re_node.metadata["uuid"])
                
                for neighbor in neighborhood:
                    # if self.G.nodes[neighbor]["type"] ==  "text_chunk":
                    #     relevant_neighbors.append(neighbor)
                    neighbor_type = self.G.nodes[neighbor]["type"]
                    edge_data = self.G.get_edge_data(node_uuid, neighbor)
                    
                    if neighbor_type == "text_chunk":
                        # Directly classify based on relation type
                        if edge_data and edge_data.get("type") == "REFERENCE_TO":
                            relevant_neighbors_referenced.append(neighbor)
                        elif edge_data and edge_data.get("type") == "SIMILAR_TO":
                            relevant_neighbors_similar.append(neighbor)
                                
                    elif neighbor_type == "keyword":
                        # Get all "text_chunk" neighbors of the keyword node
                        keyword_neighbors = self.G.neighbors(neighbor)
                        for keyword_neighbor in keyword_neighbors:
                            if self.G.nodes[keyword_neighbor]["type"] == "text_chunk":
                                keyword_edge_data = self.G.get_edge_data(neighbor, keyword_neighbor)
                                relevant_neighbors.append(keyword_neighbor)
                                
                                d2_keyword_neighbors = self.G.neighbors(keyword_neighbor)
                                
                                for d2_neighbor in d2_keyword_neighbors:
                                    d2_keyword_edge_data = self.G.get_edge_data(keyword_neighbor, d2_neighbor)
                                    if d2_keyword_edge_data and d2_keyword_edge_data.get("type") == "REFERENCE_TO":
                                        relevant_neighbors_referenced.append(d2_neighbor)
                                    elif d2_keyword_edge_data and d2_keyword_edge_data.get("type") == "SIMILAR_TO":
                                        relevant_neighbors_similar.append(d2_neighbor)
        except Exception as e:
            print(str(e))
            print(relevant_nodes[idx])
        relevant_node_set = []         
        for score, node in relevant_nodes:
            node_set = [node_s.metadata["uuid"] for score, node_s in relevant_node_set]
            
            if node.metadata["uuid"] not in node_set:
                relevant_node_set.append((score, node))
                
        r_node_set = [node_s.metadata["uuid"] for score, node_s in relevant_node_set]
        relevant_neighbors_referenced_set = list(
            set(
                [
                    neigh for neigh in relevant_neighbors_referenced if neigh not in relevant_node_set
                    ]
                )
            )
        r_node_set = r_node_set + relevant_neighbors_referenced_set
        relevant_neighbors_similar_set = list(
            set(
                [
                    neigh for neigh in relevant_neighbors_similar 
                    if neigh not in relevant_node_set and neigh not in relevant_neighbors_referenced_set
                    ]
                )
            )

        ## R_d ##
        relevant_docs = []
        for score, neighbor in relevant_node_set:
            result = self.mongo_crud.list_documents({"metadata.uuid": neighbor.metadata["uuid"]}, collection_name="Nodes")
            if len(result) > 0 and result[0].metadata["type"] == "text_chunk":
                relevant_docs.append((0,result[0]))
        
        ## R_r ##
        relevant_neighbors_referenced_docs = []
        for neighbor in relevant_neighbors_referenced:
            result = self.mongo_crud.list_documents({"metadata.uuid": neighbor}, collection_name="Nodes")
            if len(result) > 0:
                relevant_neighbors_referenced_docs.append((0,result[0]))
        
        ## R_s ## 
        relevant_neighbors_similar_docs = []
        for neighbor in relevant_neighbors_similar:
            result = self.mongo_crud.list_documents({"metadata.uuid": neighbor}, collection_name="Nodes")
            if len(result) > 0:
                relevant_neighbors_similar_docs.append((0,result[0]))
        
        ## Merge S_s and R_d ##
        context_documents = relevant_nodes + relevant_docs 
        
        
        # context_nodes = list(set([node.metadata["uuid"] for score, node in relevant_nodes] + relevant_neighbors))   
        # context_nodes = {}
        # context_documents = []
        
        # for score, re_doc in relevant_nodes + relevant_neighbor_docs:
        #     uuid = re_doc.metadata["uuid"]
        #     if re_doc.metadata["type"] != "text_chunk": 
        #         continue
            
            # context_docs = [doc.metadata["uuid"] for score, doc in context_documents]
            # if re_doc.metadata["uuid"] not in context_docs:
            #     context_documents.append((score, re_doc))
            # else:
            #     for idx, (ac_score, context_doc) in enumerate(context_documents):
            #         if context_doc == re_doc.metadata["uuid"]:
            #             context_documents[idx] = (max([ac_score, score]), re_doc)
        
        # for context_node in context_nodes:
        #     context_documents.append(self.G.nodes[context_node])
        # context_nodes = {k:v for k, v in sorted(context_nodes.items(), key=lambda item: item[1], reverse=True)}
        # for context_node, score in context_nodes.items():
        #     context_documents.append((self.G.nodes[context_node], score))
        
        # context_documents = [
        #     (score, 
        #     Document(
        #         page_content=doc["content"],
        #         metadata = {
        #             key:value for key, value in doc.items() if key is not "content"
        #         }
        #     )) for doc, score in context_documents
        # ]
        if ragflow:
            ragflow.log_step({"relevant_docs": [doc for score, doc in context_documents if score > 0]})  
            ragflow.log_step({"relevant_neighbor": [doc for score, doc in context_documents if score == 0]})    
        context = [(score, doc) for score, doc in context_documents if doc.metadata["type"] == "text_chunk"]
        neighbor_context = [doc for score, doc in context_documents if score == 0]
        return neighbor_context, context
        
    def construct_kg(self, documents:List[Document], top_k=2):
        
        ## -- Init Graph
        for doc in documents:
            doc.metadata["type"] = "text_chunk"
       
        self.G = nx.DiGraph()
        
        ## -- Add document chunks to Graph
        self.embd_model.embed_documents(documents, key="embedding")
        for doc in tqdm(documents, desc="Add text_chunks as node"):
            
            properties = {k:v for k,v in doc.metadata.items() if k not in ["uuid", "mapped_references"]}
            properties["content"] = doc.page_content
            self.G.add_node(
                doc.metadata["uuid"],
                **properties
                #type="text_chunk",
                #content = doc.page_content,
                #source = doc.metadata["source"],
                #embedding = doc.metadata["embedding"]
                
            )
            
        ## -- Extract all Keywords from Texts
        #pprint.pp([doc for doc in documents if "keywords" not in list(doc.metadata.keys())])
        keywords = []
        keywords.append([doc.metadata["keywords"] for doc in documents if "keywords" in list(doc.metadata.keys())])
        
        
        keyword_list = list(set([self.lemmatizer.lemmatize(k).lower() for keyword_list in keywords for keyword in keyword_list for k in keyword])) 
        
        keyword_docs = [
            Document(
                    page_content=keyword, 
                    metadata={"uuid": str(uuid4()),
                              "type": "keyword"}
                )
            for keyword in keyword_list]
        
        # TextRank
        # TF-IDF
        # KeyBERT
        
        ## -- Add Keyword to Graph
        
        self.embd_model.embed_documents(keyword_docs, key="embedding")
        for keyword in tqdm(keyword_docs, desc="Add keywords as nodes"):
            self.G.add_node(
                keyword.metadata["uuid"],
                type = "keyword",
                embedding = keyword.metadata["embedding"],
                content = keyword.page_content
            )
        
        ## -- Calculate the embeddings for each text
        # Encoder 
        # self.vector_store_text.add_documents(
        #     documents=documents, 
        #     ids = [doc.metadata["uuid"] for doc in documents]
        #     )
        
        
        #documents_ids=[self.mongo_crud.create_document(doc, "Text_Chunks", embedding="text_embeddings") for doc in tqdm(documents, desc="Insert documents in MongoDB")]
        #self.vector_store_text = FAISS.from_documents(documents, self.embd_model)
        
        # self.vetor_store_keywords.add_documents(
        #     documents = keyword_docs,
        #     ids = [doc.metadata["uuid"] for doc in keyword_docs]
        # )
        
        #documents_ids=[self.mongo_crud.create_document(doc, "Keywords", embedding="keyword_embeddings") for doc in tqdm(documents, desc="Insert documents in MongoDB")]
        #self.vector_store_keywords = FAISS.from_documents(keyword_docs, self.embd_model)
        
        ## -- Retrieve top-k document chunks for each keyword
        # BM25
        corpus = []
        corpus.extend([doc.page_content for doc in documents])
        corpus = []
        
        nlp = spacy.load('de_core_news_md')
        for doc in tqdm(documents, desc="Normalize text corpus"):
            sentences = nltk.sent_tokenize(doc.page_content)
            words = [nltk.word_tokenize(sentence) for sentence in sentences]
            #words = [self.hanta_ger.tag_sent(sentence, taglevel=1) for sentence in sentences]
            #corpus.append([self.lemmatizer.lemmatize(w).lower() for word in words for w in word])
            corpus.append([clean_and_normalize(w,nlp).lower() for word in words for w in word])
        #corpus.append(keywords)
        
        bm25 = CustomBM25(corpus)
        #bm25 = BM25Retriever.from_documents(documents)
        #keyword_uuid_dict = nx.get_node_attributes(self.G, "uuid")
        similarity_matrix = {}
        for doc in documents:
            doc.metadata["keyword_sim"] = {}
            
        for keyword in tqdm(keyword_docs, desc="Create Similarity Matrix Keywords/Docs"):
            scores = bm25.get_scores([keyword.page_content])
            for idx, score in enumerate(scores):
                documents[idx].metadata["keyword_sim"][keyword.metadata["uuid"]]=score

        for document in documents:
            top_k = round(3/2*len(document.page_content.split(" "))**(1/3))
            keyword_sim = {k:v for k, v in sorted(document.metadata["keyword_sim"].items(), key=lambda item: item[1], reverse=True)[:top_k]}
            for keyword, score in keyword_sim.items():
                #keyword, score = keyword_score
                text_node = document.metadata["uuid"]
                self.G.add_edge(
                    u_of_edge = text_node, 
                    v_of_edge = keyword,
                    type="HAS_KEYWORD",
                    weight=float(score)
                    )

        # for keyword in tqdm(keyword_docs, desc="Match keyword to text-chunk with BM25"):
        #     scores = bm25.get_scores([keyword.page_content])
        #     # idf = bm25.idf.get("ATW")
        #     #scores = bm25.invoke("ATW")
        #     ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        #     uuid = keyword.metadata["uuid"]
        #     keyword_node = self.G.nodes[uuid]
            
        #     for doc_index, score in ranked_docs:
        #         text_node = self.G.nodes[documents[doc_index].metadata["uuid"]]
        #         self.G.add_edge(
        #             u_of_edge = documents[doc_index].metadata["uuid"], 
        #             v_of_edge = uuid,
        #             type="HAS_KEYWORD" 
        #             #weight=float(score)
        #             )
        
        # for document in documents:
        #     references = document.metadata["mapped_references"]
            
        #     for reference in references:
        #         key_names = reference.keys()
        #         if "document_name" in key_names:
        #             reference_name = reference["document_name"]
                    
        #             for doc in documents:
                        
        #                 source_doc = doc.metadata["source"]
        #                 print(str(reference_name), str(source_doc))
        #                 if reference_name in source_doc:
        #                     print("<--- it's a match ---> ")
                        
                    
            
                
        for document in tqdm(documents, desc="Find relevant references"):
            references = document.metadata["mapped_references"]
            for ref in references:
                if "document_name" in list(ref.keys()):
                    ref_docs = []
                    for doc in documents:
                        ref_doc = ref["document_name"]
                        source = doc.metadata["source"]
                        print(str(ref_doc), " --- ", str(source))
                        if ref_doc in source \
                            and doc.metadata["uuid"] != document.metadata["uuid"]:
                                ref_docs.append(doc)
                    #ref_docs = [doc for doc in documents if ref["document_name"] in doc.metadata["source"] and doc.metadata["uuid"] != document.metadata["uuid"]]
                    if len(ref_docs) > 0:
                        doc_embd = document.metadata["embedding"]
                        ref_embd = [ref.metadata["embedding"] for ref in ref_docs]
                        similarities = util.cos_sim(doc_embd, ref_embd)[0]
                        similarity_scores = [(score, ref_docs[idx]) for idx, score in enumerate(similarities)]
                        sim_top_k = 3 if len(similarity_scores) > 3 else (len(similarity_scores)-1)
                        similar_docs = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:sim_top_k]
                        
                        for score, similar_doc in similar_docs:
                            self.G.add_edge(
                                u_of_edge=document.metadata["uuid"],
                                v_of_edge=similar_doc.metadata["uuid"],
                                type="REFERENCE_TO",
                                weight = score.item()
                            )
                    #continue ## delete it - only for debugging
        removing_nodes = []
        for node in self.G.nodes:
            if self.G.nodes[node].get("type") == "keyword" and \
                self.G.degree(node) == 0:
                    for idx, keyword in enumerate(keyword_docs):
                        if keyword.metadata["uuid"] == node:
                            removing_nodes.append(node)
                            keyword_docs.pop(idx)    
        
        self.G.remove_nodes_from(removing_nodes)   
        # for node in self.G.nodes():
        
        # for document in documents:
        #     exclude_docs = [doc for doc in documents if doc.metadata["uuid"] != document.metadata["uuid"]]
        #     a_embedding = document.metadata["embedding"]
        #     b_embeddings = [doc.metadata["embedding"] for doc in exclude_docs]
        #     embedding_scores = util.cos_sim(a_embedding, b_embeddings)[0]
        #     sorted_embedding_idx = np.argsort(embedding_scores)[-10:]
        #     knn_docs = [documents[idx] for idx in sorted_embedding_idx if embedding_scores[idx] > 0.75]

        #     for idx in sorted_embedding_idx:
        #         if embedding_scores[idx] > 0.75:
        #             self.G.add_edge(
        #                         u_of_edge=document.metadata["uuid"],
        #                         v_of_edge=exclude_docs[idx].metadata["uuid"],
        #                         type="SIMILAR_TO",
        #                         weight = embedding_scores[idx].item()
        #                     )
        #     self.G.nodes[node]["text_embd"] = self.embd_model.embed_text(self.G.nodes[node]["content"])
            
        
        ## -- Construct Egograph for each Keyword
        # Calculate the node embeddings 
        # store the nodes embeddings with node in vector db
        # node_uuid : embd_vector
        
        #self.embd_model.embed_documents(keyword_docs, key="keyword_embeddings")
        #nodes = documents + keyword_docs
        documents_ids=[self.mongo_crud.create_document(doc, "Nodes", embedding="embedding", ) for doc in tqdm(documents, desc="Insert documents in MongoDB")]
        keywords_ids=[self.mongo_crud.create_document(doc, "Nodes", embedding="embedding") for doc in tqdm(keyword_docs, desc="Insert keywords in MongoDB")]

        #self.vector_store_nodes = FAISS.from_documents(keyword_docs + documents, self.embd_model)
        self.node_list = {}
        #self.vector_store_nodes.add_documents(documents, ids=[doc.metadata["uuid"] for doc in documents])
        for node in tqdm(self.G.nodes(), desc="Calculate graph node embeddings"):
            graph_node_embeddings = self.GraphSAGE.calculate(
                                                            node=node,
                                                            nx_graph=self.G,
                                                            search_depth=2
                                                        )
            
            node_data = self.G.nodes[node] #["graph_embd"]
            self.node_list[node] = node_data
            self.mongo_crud.update_documents_by_uuid(
                uuid_value={"metadata.uuid": node}, 
                updated_fields={
                    "graph_embd": self.G.nodes[node]["graph_embd"].tolist(),
                    "neighbor_embeddings": self.G.nodes[node]["neighbor_embd"].tolist()
                    }, 
                collection_name="Nodes"
                ) 
            
        try:
        # Insert the NetworkX graph into Neo4j
            self.neo4j_inserter.insert_networkx_graph(self.G)
            print("NetworkX graph has been successfully inserted into Neo4j!")
        finally:
            # Always close the driver connection when done.
            self.neo4j_inserter.close()
            #node_attr = nx.get_node_attributes(self.G)
            # doc = Document(
            #             page_content=node,
            #         )
            #self.vector_store_nodes.add_documents([doc])   
    
    