from typing import Dict, List
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from neo4j import GraphDatabase

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


class Neo4jGraphInserter:
    """
    A class to insert a NetworkX graph into a Neo4j database.
    """
    def __init__(self, uri, username, password):
        """
        Initialize the Neo4j driver connection.

        :param uri: URI for the Neo4j database (e.g., "bolt://localhost:7687")
        :param username: Neo4j username
        :param password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()

    def insert_networkx_graph(self, graph):
        """
        Insert nodes and relationships from a NetworkX graph into the Neo4j database.

        :param graph: A NetworkX graph object.
        """
        def create_nodes_and_relationships(tx):
            # --- Insert nodes with dynamic labels ---
            for node_id, properties in graph.nodes(data=True):
                # Copy properties to avoid mutating the original data.
                node_props = properties.copy()
                # Determine the node's label. If not specified, use "Node" as default.
                node_label = node_props.pop("type", "Node")
                
                # Build the query string dynamically with the node label.
                query = f"""
                    MERGE (n:{node_label} {{id: $id}})
                    SET n += $props
                """
                tx.run(query, id=node_id, props=node_props)
            
            # --- Insert relationships with dynamic types ---
            for source, target, properties in graph.edges(data=True):
                # Copy properties to avoid mutating the original data.
                edge_props = properties.copy()
                # Determine the relationship type. If not specified, use "CONNECTED_TO" as default.
                rel_type = edge_props.pop("type", "CONNECTED_TO")
                
                # Build the query string dynamically with the relationship type.
                query = f"""
                    MATCH (a {{id: $source_id}})
                    MATCH (b {{id: $target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += $props
                """
                tx.run(query, source_id=source, target_id=target, props=edge_props)
                
        with self.driver.session() as session:
            session.write_transaction(create_nodes_and_relationships)

# Example usage
class MockDocument:
    """Mock class for a Langchain Document."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class NetworkXImporter:
    def __init__(self):
        # Initialize an in-memory graph using NetworkX
        self.graph = nx.DiGraph()

    # Function to get or create a node
    def get_or_create_node(self, node_id, label, **attributes):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, label=label, **attributes)
        return self.graph.nodes[node_id]

    # Function to add an edge between two nodes with a given relationship type
#     def add_edge(self, from_node, to_node, relationship):
#         self.graph.add_edge(from_node, to_node, relationship=relationship)
        
    def add_edge(self, from_node, to_node, **kwargs):
        self.graph.add_edge(from_node, to_node, **kwargs)

    # Function to process a single document's text chunk
    def process_document_chunk(self, doc):
        # Extract metadata
        print(doc)
        source_doc = doc.metadata.get("source")
        headers = {f"Header{i}": doc.metadata.get(f"Header{i}") for i in range(1, 5)}
        referenced_docs = doc.metadata.get("referenced_documents", [])
        keywords = doc.metadata.get("keywords", [])
        uuid = doc.metadata.get("uuid")
        previous_uuid = doc.metadata.get("previous_uuid")
        content = doc.page_content

        if not uuid:
            raise ValueError("Each text chunk must have a 'uuid'.")

        # Get or create the root node for the source document
        self.get_or_create_node(source_doc, "Document", name=source_doc)

        # Create or retrieve the current text chunk node
        self.get_or_create_node(
            uuid,
            "TextChunk",
            uuid=uuid,
            text=content,
            **headers,
            source_document=source_doc
        )

        # Add an edge from the document to the text chunk
        self.add_edge(source_doc, uuid, relationship="HAS_TEXT")

        # Link the text chunk to referenced documents
        for ref_doc in referenced_docs:
            ref_name = ref_doc.get("name")
            ref_revision = ref_doc.get("revision")
            ref_last_update = ref_doc.get("last_update")

            if not ref_name:
                continue  # Skip if no name is provided for the referenced document

            # Create a unique identifier for the referenced document node
            ref_node_id_parts = [ref_name]
            if ref_revision:
                ref_node_id_parts.append(ref_revision)
            elif ref_last_update:
                ref_node_id_parts.append(ref_last_update)
            
            ref_node_id = ref_name
            #ref_node_id = "-".join(ref_node_id_parts)

            # Get or create the referenced document node
            self.get_or_create_node(
                ref_node_id,
                "Document",
                name=ref_name,
                revision=ref_revision,
                last_update=ref_last_update,
                description=ref_doc.get("description")
            )

            # Add an edge from the text chunk to the referenced document
            self.add_edge(uuid, ref_node_id, relationship="REFERENCES")

        # Link the text chunk to keywords
        for keyword in keywords:
            keyword_node = self.get_or_create_node(keyword, "Keyword", name=keyword)
            self.add_edge(uuid, keyword, relationship="HAS_KEYWORD")

        # Link to the previous text chunk (if it exists and belongs to the same document)
        if previous_uuid:
            previous_chunk_node = self.graph.nodes.get(previous_uuid)
            if previous_chunk_node and previous_chunk_node.get("source_document") == source_doc:
                self.add_edge(uuid, previous_uuid, relationship="PREVIOUS")
     
    # Function to add similarity edges after processing
    def add_similarity_edges(self, documents, k=5, model_name='all-MiniLM-L6-v2'):
        """
        Adds edges between nodes based on similarity using KNN.

        Parameters:
        - passages (list of str): List of all text passages (document chunks).
        - k (int): Number of nearest neighbors to connect.
        - model_name (str): Name of the Sentence Transformers model.
        """
        passages = [doc.page_content for doc in documents]
        knn_graph, similarity_matrix = construct_knn_graph(passages, model_name=model_name, k=k)
        print("KNN_Graph: ", knn_graph)
        print("Sim_Matrix: ", similarity_matrix)
        
        for idx, doc in enumerate(documents):
            doc.metadata["knn"] = {document.metadata["uuid"]:similarity_matrix[idx][i] for i, document in enumerate(documents)}
            print(doc)
    
        #return
        
        for from_node in documents:
            for to_node, weight in from_node.metadata["knn"].items():
                if from_node.metadata["uuid"] != to_node:
                    self.add_edge(from_node.metadata["uuid"], to_node, relationship="SIMILARITY", weight=weight)
           
        # for edge in knn_graph.edges(data=True):
        #     from_node, to_node = edge[0], edge[1]
        #     weight = edge[2]['weight']
        #     self.add_edge(from_node, to_node, relationship="SIMILARITY", weight=weight)

    # Function to process multiple documents
    def process_documents(self, doc_list: List):
        for doc in doc_list:
            self.process_document_chunk(doc)

    # Function to print the graph structure visually
    def print_graph(self, output_html="graph.html", output_svg="graph.svg"):
        # Get positions for the nodes using a layout algorithm
        pos = nx.spring_layout(self.graph)

        # Extract data for plotting
        edge_x = []
        edge_y = []
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_labels = []
        for node, data in self.graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if data["type"] == "keyword":
                label = f"{data['type']}: {data["content"]}"
            else:
                label = f"{data['type']}: {node}"
            node_labels.append(label)

        # Edge traces
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Node traces
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=10,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    #titleside="right",
                ),
            ),
        )

        # Layout and figure setup
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Graph Visualization",
                #titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            ),
        )

        # Export to HTML
        pio.write_html(fig, file=output_html)
        print(f"Interactive graph saved as: {output_html}")

        # Export to SVG
        pio.write_image(fig, file=output_svg, format="svg", engine="kaleido")
        print(f"Static graph saved as: {output_svg}")

        # Show the graph in the notebook (optional)
        fig.show()
        
