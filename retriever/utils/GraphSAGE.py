import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np
import networkx as nx


class GraphSAGELayer(nn.Module):
    """
    A single GraphSAGE layer that updates a node’s feature by combining
    its own features with an aggregated representation of its neighbors.
    
    The update rule is:
        h_v^(k) = ReLU( W * [h_v^(k-1) || agg({h_u^(k-1) for u in N(v)})] )
    """
    def __init__(self, hidden_size, aggregator_type="mean", dropout=0.5):
        super(GraphSAGELayer, self).__init__()
        self.aggregator_type = aggregator_type
        # Input dimension is 2*hidden_size because we are concatenating the node features
        # with the aggregated neighbor features.
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, self_feats, aggregated_neigh_feats):
        # Concatenate node features with aggregated neighbor features.
        h = torch.cat([self_feats, aggregated_neigh_feats], dim=1)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h

class GraphSAGEEmbeddings(nn.Module):
    """
    A GraphSAGE model that uses a pretrained encoder to generate initial node embeddings,
    and then refines them using GraphSAGE layers.
    
    The resulting node embeddings can be used for similarity calculations.
    
    Args:
        bert_model_name (str): Pretrained encoder model name.
        num_layers (int): Number of GraphSAGE layers.
        aggregator_type (str): Type of neighbor aggregator (currently "mean" is implemented).
        dropout (float): Dropout rate.
    """
    def __init__(self, embeddings_model, depth=2, aggregator_type="mean"):
        self.embd_model = embeddings_model
        self.search_depth = depth
        self.aggregator_type = aggregator_type

    def calculate(self, 
                  node,
                  nx_graph, 
                  search_depth=2):
        # Find all nodes without GraphSAGE embeddings
        
        #missing_nodes = self.find_nodes_missing_graphsage_embd(nx_graph)
        

        #for node in missing_nodes:
            # Calculate the embeddings for all nodes
            # Get all nodes within 'search_depth' hops from the node using BFS.
            # This returns a dict: {node: distance, ...}
        neighbors_dict = nx.single_source_shortest_path_length(nx_graph, node, cutoff=search_depth)
        
        # Remove the source node itself (distance 0).
        if node in neighbors_dict:
            del neighbors_dict[node]
        
        # The keys of the dictionary are the neighbor node IDs.
        neighbor_nodes = list(neighbors_dict.keys())
        
        neighbor_embeddings = []
        
        
        for neighbor in neighbor_nodes:
            neighbor_attr = nx_graph.nodes[neighbor]
            if neighbor_attr.get("type") in ["text_chunk", "keyword"] \
                and "embedding" in neighbor_attr:
                    neighbor_embeddings.append(neighbor_attr["embedding"])
            else:
                if "embedding" not in neighbor_attr:
                    text=nx_graph.nodes[neighbor]["content"]
                    nx_graph.nodes[neighbor]["embedding"] = self.embd_model.embed_text(text)
                    neighbor_attr = nx_graph.nodes[neighbor]
                    neighbor_embeddings.append(neighbor_attr["embedding"])
        
            # Calculate the mean embedding if there are valid neighbor embeddings.
        if neighbor_embeddings:
            neighborhood_mean = np.mean(neighbor_embeddings, axis=0)
        else:
            # Option 1: Set a default embedding, e.g., a zero vector.
            # You might need to define the correct dimension (e.g., 768 for BERT).
            emb_dim = self.embd_model.hidden_size  # Adjust as needed.
            neighborhood_mean = np.zeros(emb_dim)
            # Option 2: Alternatively, you could set neighborhood_mean = None.
        
        # Add or update the 'graphsage_embd' property for the node.
        nx_graph.nodes[node]["neighbor_embd"] = neighborhood_mean
        text_embeddings = nx_graph.nodes[node]["embedding"]
        node_embeddings = neighborhood_mean + text_embeddings
        nx_graph.nodes[node]["graph_embd"] = node_embeddings
        
        return node_embeddings

    def find_nodes_missing_graphsage_embd(self, graph):
        """
        Finds all nodes in the graph that either do not have the 'graphsage_embd' property 
        or have it set to an empty/falsy value.
        
        Args:
            graph (networkx.Graph): The input graph with node attributes.
            
        Returns:
            list: A list of node identifiers that are missing a valid 'graphsage_embd'.
        """
        missing_nodes = []
        
        # Iterate over nodes along with their attribute dictionaries.
        for node, attr in graph.nodes(data=True):
            # Check if the property is missing or evaluates to False (empty, None, etc.)
            if "graphsage_embd" not in attr or not attr["graphsage_embd"]:
                missing_nodes.append(node)
                
        return missing_nodes

