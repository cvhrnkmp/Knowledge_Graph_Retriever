from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from langchain.docstore.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from collections import deque
# Define a function to create embeddings and construct KNN graph
def construct_knn_graph(passages, model_name='all-MiniLM-L6-v2', k=5):
    """
    Constructs a KNN graph based on passage embeddings.

    Parameters:
    - passages (list of str): List of text passages for which embeddings will be computed.
    - model_name (str): Name of the Sentence Transformers model.
    - k (int): Number of neighbors for KNN graph.

    Returns:
    - knn_graph (networkx.Graph): The KNN graph constructed.
    - similarity_matrix (np.ndarray): Pairwise similarity matrix.
    """
    
    #passages = [(doc.metadata["uuid"], doc.page_content) for doc in documents]
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings for each passage
    embeddings = model.encode(passages, convert_to_tensor=False)

    # Compute the pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Construct a KNN adjacency matrix using scikit-learn
    adjacency_matrix = kneighbors_graph(embeddings, n_neighbors=k, mode='connectivity', include_self=False).toarray()

    # Create a weighted graph using NetworkX
    knn_graph = nx.Graph()

    # Add edges with weights based on similarity
    num_nodes = len(passages)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] > 0:  # Check if nodes are neighbors
                weight = similarity_matrix[i, j]
                knn_graph.add_edge(i, j, weight=weight)

    return knn_graph, similarity_matrix

def find_knn_from_documents(documents:Document, k=3, inplace=False):
    passages = [doc.page_content for doc in documents]
    graph, sim_matrix = construct_knn_graph(passages, k=k)
    
    if inplace:
        for i, similarity in enumerate(sim_matrix):
            knn_list = {documents[j].metadata["uuid"] : similarity[j] for j, _ in enumerate(documents) if i!=j}
            knn_list = dict(sorted(knn_list.items(), key=lambda x: x[1], reverse=True)[:k])
            documents[i].metadata["knn"] = knn_list
            
    return graph, sim_matrix

def extract_keywords_from_documents(documents, inplace = False):
    """Extracts key terms using TF-IDF from a list of documents."""
    titles = [doc.metadata["source"] for doc in documents]
    texts = [doc.page_content for doc in documents]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    keywords_per_doc = []
    for i, title in enumerate(titles):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        sorted_indices = tfidf_scores.argsort()[::-1]
        top_keywords = [feature_names[idx] for idx in sorted_indices[:10]]  # Top 10 keywords
        keywords_per_doc.append((title, top_keywords))
        
        if inplace:
            documents[i].metadata["tf-idf-keywords"] = top_keywords + title
        
    return keywords_per_doc

def tfidf_search(V, X, query, documents):
    """Performs TF-IDF based search to retrieve relevant document nodes."""
    _, tfidf_matrix, uuids, vectorizer = extract_keywords_from_documents(documents)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    # Retrieve top relevant UUIDs
    top_k = 5  # Retrieve top 5 relevant passages
    top_uuids = [uuids[idx] for idx in ranked_indices[:top_k] if similarity_scores[idx] > 0.1]
    
    return top_uuids


def llm_based_kg_traversal(question, documents, G, fGT, K, g):
    """LLM-based KG Traversal Algorithm to Retrieve Relevant Context."""
    V = list(G.nodes)
    X = {node: G.nodes[node].get('type', '') for node in V}
    
    # Step 1: Initialize seed passages
    Vs = g(V, X, question, documents)
    
    # Step 2: Initialize the retrieved passage queue
    P = deque([{vi} for vi in Vs])
    
    # Step 3: Initialize the candidate neighbor queue
    C = deque([list(G.neighbors(vi)) for vi in Vs])
    
    # Step 4: Initialize the retrieved passage counter
    k = sum(len(Pi) for Pi in P)
    
    # Step 5: Perform graph traversal
    while P and C:
        Pi = P.popleft()
        Ci = C.popleft()
        
        # Step 7: Graph Traversal using LLM-guided function
        V_prime_i = fGT({question} | Pi, Ci, k)
        
        for v in V_prime_i:
            P.append(Pi | {v})
            C.append(list(G.neighbors(v)))
            k += 1
            if k > K:
                return P
    
    return P