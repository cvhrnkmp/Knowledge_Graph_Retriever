import os
from typing import Dict, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain.docstore.document import Document

# Load environment variables from .env file if it exists
load_dotenv()

# Default MongoDB configuration
DEFAULT_MONGODB_CONFIG = {
    'username': os.getenv('MONGODB_USERNAME', 'user'),
    'password': os.getenv('MONGODB_PASSWORD', 'pass'),
    'cluster': os.getenv('MONGODB_CLUSTER', 'localhost'),
    'port': os.getenv('MONGODB_PORT', '27019'),
    'database': os.getenv('MONGODB_DATABASE', 'From_Local_to_Global'),
    'auth_source': os.getenv('MONGODB_AUTH_SOURCE', 'admin'),
    'auth_mechanism': os.getenv('MONGODB_AUTH_MECHANISM', 'SCRAM-SHA-1')
}

class MongoCRUD:
    
    def __init__(self, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None, 
                 cluster_url: Optional[str] = None, 
                 database_name: str = None, 
                 default_collection_name: str = None,
                 connection_string: Optional[str] = None):
        """
        Initialize the MongoCRUD instance using either a connection string or individual parameters.
        If no parameters are provided, uses environment variables with defaults.

        Args:
            username (str, optional): MongoDB username. Defaults to MONGODB_USERNAME env var or 'user'.
            password (str, optional): MongoDB password. Defaults to MONGODB_PASSWORD env var or 'pass'.
            cluster_url (str, optional): The cluster URL. Defaults to MONGODB_CLUSTER env var or 'localhost'.
            database_name (str, optional): Name of the database. Defaults to MONGODB_DATABASE or 'From_Local_to_Global'.
            default_collection_name (str, optional): Default collection name. Required if not using connection_string.
            connection_string (str, optional): Complete MongoDB connection string. If provided, other params are ignored.
        """
        if connection_string is None:
            # Use provided parameters or fall back to environment variables with defaults
            config = {
                'username': username or DEFAULT_MONGODB_CONFIG['username'],
                'password': password or DEFAULT_MONGODB_CONFIG['password'],
                'cluster': cluster_url or DEFAULT_MONGODB_CONFIG['cluster'],
                'port': DEFAULT_MONGODB_CONFIG['port'],
                'auth_source': DEFAULT_MONGODB_CONFIG['auth_source'],
                'auth_mechanism': DEFAULT_MONGODB_CONFIG['auth_mechanism']
            }
            
            if not database_name:
                database_name = DEFAULT_MONGODB_CONFIG['database']
            
            if not default_collection_name:
                raise ValueError("default_collection_name is required when not using connection_string")
                
            connection_string = (
                f"mongodb://{config['username']}:{config['password']}@{config['cluster']}:{config['port']}/"
                f"?retryWrites=true&w=majority&directConnection=true"
                f"&authSource={config['auth_source']}&authMechanism={config['auth_mechanism']}"
            )
        
        self.client = MongoClient(connection_string)
        self.database = self.client[database_name] if database_name else self.client.get_database()
        self.default_collection_name = default_collection_name
        self.collection = self.database[default_collection_name] if default_collection_name else None
        
        print(f"Connected to MongoDB database: {self.database.name}" + 
              (f", default collection: {default_collection_name}" if default_collection_name else ""))

    def _get_collection(self, collection_name: str = None):
        """Helper method to retrieve the desired collection."""
        if collection_name:
            return self.database[collection_name]
        return self.collection

    def create_document(self, document: Document, collection_name: str = None, embedding = None) -> str:
        """
        Insert a LangChain Document into the specified collection.

        Args:
            document (Document): A LangChain Document.
            embedding (list, optional): A list of floats representing the document's embedding.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            str: The string representation of the inserted document's _id.
        """
        coll = self._get_collection(collection_name)
        data = {
            "page_content": document.page_content,
            "metadata": document.metadata
        }
        if embedding is not None:
            data["embedding"] = document.metadata[embedding]
        result = coll.insert_one(data)
        print(f"Inserted document with _id: {result.inserted_id} into collection: {coll.name}")
        return str(result.inserted_id)

    def insert(self, data:Dict, collection_name: str = None) -> str:
        """
        Insert a LangChain Document into the specified collection.

        Args:
            document (Document): A LangChain Document.
            embedding (list, optional): A list of floats representing the document's embedding.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            str: The string representation of the inserted document's _id.
        """
        coll = self._get_collection(collection_name)
        result = coll.insert_one(data)
        print(f"Inserted document with _id: {result.inserted_id} into collection: {coll.name}")
        return str(result.inserted_id)
    
    def read_document(self, doc_id: str, collection_name: str = None) -> Document:
        """
        Retrieve a document by its _id from the specified collection.

        Args:
            doc_id (str): The _id of the document as a string.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            Document: The LangChain Document if found; otherwise, None.
        """
        coll = self._get_collection(collection_name)
        data = coll.find_one({"_id": ObjectId(doc_id)})
        if data:
            print(f"Found document in collection: {coll.name}:\n{data}")
            return Document(page_content=data.get("page_content"), metadata=data.get("metadata"))
        else:
            print("No document found with the provided _id.")
            return None

    def update_document(self, doc_id: str, updated_fields: dict, collection_name: str = None) -> bool:
        """
        Update fields of a document by its _id in the specified collection.

        Args:
            doc_id (str): The _id of the document as a string.
            updated_fields (dict): Dictionary of fields to update.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            bool: True if at least one field was updated, False otherwise.
        """
        coll = self._get_collection(collection_name)
        result = coll.update_one({"_id": ObjectId(doc_id)}, {"$set": updated_fields})
        if result.modified_count > 0:
            print(f"Updated document {doc_id} with fields: {updated_fields} in collection: {coll.name}")
            return True
        else:
            print("No documents were updated.")
            return False
        
    def upsert(self, data:Dict, collection_name: str = None) -> bool:
        """
        Update fields of a document by its _id in the specified collection.

        Args:
            doc_id (str): The _id of the document as a string.
            updated_fields (dict): Dictionary of fields to update.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            bool: True if at least one field was updated, False otherwise.
        """
        coll = self._get_collection(collection_name)
        result = coll.update_one(data, {"$set": data}, upsert=True)
        if result.modified_count > 0:
            #print(f"Updated document {doc_id} with fields: {updated_fields} in collection: {coll.name}")
            return True
        else:
            #print("No documents were updated.")
            return False
        
    def check_existens(self, data:Dict, collection_name: str = None) -> bool:
        """
        Update fields of a document by its _id in the specified collection.

        Args:
            doc_id (str): The _id of the document as a string.
            updated_fields (dict): Dictionary of fields to update.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            bool: True if at least one field was updated, False otherwise.
        """
        coll = self._get_collection(collection_name)
        result = coll.find_one(data)
        if result:
            #print(f"Updated document {doc_id} with fields: {updated_fields} in collection: {coll.name}")
            return True
        else:
            #print("No documents were updated.")
            return False

    def delete_document(self, doc_id: str, collection_name: str = None) -> bool:
        """
        Delete a document by its _id from the specified collection.

        Args:
            doc_id (str): The _id of the document as a string.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            bool: True if a document was deleted, False otherwise.
        """
        coll = self._get_collection(collection_name)
        result = coll.delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count > 0:
            print(f"Deleted document with _id: {doc_id} from collection: {coll.name}")
            return True
        else:
            print("No documents were deleted.")
            return False

    def get_all_documents(self, filter:dict= {}, collection_name:str = None, as_lg_document = True) -> list:
        coll = self._get_collection(collection_name=collection_name)
        cursor = coll.find(filter)
        if as_lg_document:
            documents = []
            for data in cursor:
                doc = (Document(
                    page_content=data.get("page_content"), 
                    metadata=data.get("metadata"))
                                )
                doc.metadata["embedding"] = data.get("embedding")
                doc.metadata["graph_embd"] = data.get("graph_embd")
                doc.metadata["neighbor_embeddings"] = data.get("neighbor_embeddings")
                documents.append(doc)
            return documents
        else:
            return list(cursor)
        #print(f"Retrieved {len(documents)} documents from collection: {coll.name}")
        return documents
        
    def list_documents(self, query: dict = None, collection_name: str = None) -> list:
        """
        Retrieve multiple documents matching a query from the specified collection.

        Args:
            query (dict, optional): MongoDB query dictionary. Defaults to {}.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            list: A list of LangChain Document objects.
        """
        if query is None:
            query = {}
        coll = self._get_collection(collection_name)
        cursor = coll.find(query)
        documents = []
        for data in cursor:
            documents.append(Document(page_content=data.get("page_content"), metadata=data.get("metadata")))
        #print(f"Retrieved {len(documents)} documents from collection: {coll.name}")
        return documents

    # --- Vector Search (Cosine Similarity / kNN) using MongoDB Atlas Search ---

    def cosine_similarity_search(self, query_vector: list, top_k: int = 5, collection_name: str = None, index_name: str = "default", path:str="embedding", filter:dict=None) -> list:
        """
        Perform a cosine similarity (kNN) search using MongoDB Atlas Search.
        This method uses the `$search` aggregation stage with the `knnBeta` operator,
        which leverages your vector search index. It returns the top_k documents
        ranked by similarity to the provided query_vector.

        **Note:** Ensure that the vector search index is created on the target collection,
        and that your documents include an "embedding" field (e.g., a normalized vector).

        Args:
            query_vector (list): The query embedding (list of floats).
            top_k (int, optional): The number of top similar documents to return. Defaults to 5.
            collection_name (str, optional): The collection to use. Defaults to the default collection.
            index_name (str, optional): The name of the Atlas Search index to use. Defaults to "default".

        Returns:
            list: A list of LangChain Document objects corresponding to the top similar documents.
        """
        
        coll = self._get_collection(collection_name)
        pipeline = [
            {
                '$vectorSearch': {
                    'index': index_name, 
                    'path': path, 
                    'queryVector': query_vector.tolist(),
                    'exact': True,
                    'limit': top_k,
                }
            }, 
            {
                '$project': {
                '_id': 0, 
                'page_content': 1, 
                'metadata': 1, 
                'references': 1,
                'keywords': 1,
                'type': 1,
                'embedding': 1,
                'graph_embd': 1,
                'neighbor_embeddings': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
                }
            }
            ]
        if filter:
            pipeline[0]["$vectorSearch"]["filter"] = filter
        results = list(coll.aggregate(pipeline))
        docs = []
        for doc in results:
            lc_doc = Document(page_content=doc.get("page_content"), metadata=doc.get("metadata"))
            # lc_doc.metadata["references"] = doc.get("metadata.references")
            # lc_doc.metadata["keywords"] = doc.get("metadata.keywords")
            # lc_doc.metadata["type"] = doc.get("metadata.type")
            lc_doc.metadata["embedding"] = doc.get("embedding")
            lc_doc.metadata["graph_embd"] = doc.get("graph_embd")
            lc_doc.metadata["neighbor_embeddings"] = doc.get("neighbor_embeddings")
            cosine_similarity = doc.get("score")*2 - 1 
            docs.append((cosine_similarity, lc_doc))
        return docs
    
    def update_documents_by_uuid(self, uuid_value: dict, updated_fields: dict, collection_name: str = None) -> bool:
        """
        Update all documents in the specified collection that have a specific "uuid" field.

        Args:
            uuid_value (str): The UUID value to match in documents.
            updated_fields (dict): Dictionary of fields to update.
            collection_name (str, optional): The collection to use. Defaults to the default collection.

        Returns:
            bool: True if at least one document was updated, False otherwise.
        """
        coll = self._get_collection(collection_name)
        result = coll.update_many(uuid_value, {"$set": updated_fields})
        if result.modified_count > 0:
            #print(f"Updated {result.modified_count} documents with uuid: {uuid_value} with fields: {updated_fields} in collection: {coll.name}")
            return True
        else:
            print(f"No documents with uuid: {uuid_value} were updated in collection: {coll.name}")
            return False
# if __name__ == "__main__":
#     # Replace these with your actual MongoDB Atlas connection details.
#     USERNAME = "user"
#     PASSWORD = "pass"
#     CONNECTION_STRING = "localhost"
#     DATABASE_NAME = "Connecting_the_Dots"
#     DEFAULT_COLLECTION_NAME = "Text_Chunks"

#     # Initialize the CRUD handler with a default collection.
#     mongo_crud = MongoCRUD(USERNAME, PASSWORD, CONNECTION_STRING, DATABASE_NAME, DEFAULT_COLLECTION_NAME)

#     # You can also specify a different collection for a particular operation:
#     specific_collection = "Text_Chunks"

#     # Create a new LangChain document with an embedding in the specific collection.
#     new_doc = Document(page_content="This is a test document.", metadata={"source": "test", "embeddings":[0.1, 0.2, 0.3]})
#     inserted_id = mongo_crud.create_document(new_doc, collection_name=specific_collection, embedding = "embeddings")

#     # Read the document back from the specific collection.
#     retrieved_doc = mongo_crud.read_document(inserted_id, collection_name=specific_collection)
#     print("Retrieved Document:", retrieved_doc)

#     # # Update the document's metadata in the specific collection.
#     # updated = mongo_crud.update_document(inserted_id, {"metadata.source": "updated_test"}, collection_name=specific_collection)
#     # print("Document updated:", updated)

#     # # List all documents in the specific collection.
#     # all_docs = mongo_crud.list_documents(collection_name=specific_collection)
#     # for doc in all_docs:
#     #     print(doc)

#     # # Perform a cosine similarity search using the Atlas Search vector function in the default collection.
#     # query_embedding = [0.1, 0.25, 0.35]
#     # similar_docs = mongo_crud.cosine_similarity_search(query_embedding, top_k=3, index_name="default")
#     # print("Cosine Similarity Search Results:")
#     # for doc in similar_docs:
#     #     print(doc)

#     # Delete the document from the specific collection.
#     #deleted = mongo_crud.delete_document(inserted_id, collection_name=specific_collection)
#     #print("Document deleted:", deleted)