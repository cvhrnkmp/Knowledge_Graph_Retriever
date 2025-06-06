# Knowledge Graph Retriever

A flexible FastAPI service to query and explore knowledge graphs using multi-step retrieval and LLM-powered reasoning. Users can dynamically choose chat and embedding models, as well as retrieval strategies (e.g., vector search, BM25, HyDE, graph traversal) on each request.

---

## Features

- **Dynamic model selection**: Specify `chat_model` and `embeddings_model` per request.
- **Pluggable retrievers**: Support for vector search, BM25 keyword search, HyDE, and graph-based traversals.
- **Multi-stage DRIFT pipeline**: Expand, decompose, retrieve, and reduce queries with logging.
- **MongoDB-backed storage**: Fetch documents, entities, relations, and communities from MongoDB.
- **Structured JSON responses**: Return answers with supporting context.

---

## Installation

```bash
git clone https://your.repo.url/Knowledge_Graph_Retriever_API.git
cd Knowledge_Graph_Retriever_API
# using Poetry
poetry install
# or with pip
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with your connection settings:

```dotenv
MONGODB_USERNAME=your_mongo_user
MONGODB_PASSWORD=your_mongo_pass
MONGODB_CLUSTER=your_mongo_host
MONGODB_DATABASE=your_db_name
```

---

## Running the API

Start the server in development:

```bash
poetry run uvicorn api:app --reload
# or
uvicorn api:app --reload
```

By default, the server listens on `http://127.0.0.1:8000`.

---

## Usage Example

Send a `POST` request to `/search` with a JSON body:

```json
{
  "global_query": "What causes community drift in social networks?"
}
```

Response:

```json
"response": str
```

### Available Search Types

- `vector`: Vector similarity search
- `bm25`: BM25 keyword-based search
- `hyde`: Hypothetical Document Embeddings
- `connecting_dots`: Graph-based traversal

## Configuration

### Retriever Configuration

You can configure different retrieval strategies in `traversing_process.py`. The `DRIFTSearch` class accepts the following parameters:

- `chat_model`: Name of the chat model to use (default: "llama3.2")
- `embeddings_model`: Name of the embeddings model to use (default: "llama3.2")
- `dataset`: Name of the dataset to search against

### Search Initialization

The `init_search` method allows you to configure the search strategy for different components:

```python
searcher.init_search(
    community_search="vector",    # Search strategy for communities
    entity_search="vector",       # Search strategy for entities
    relation_search="vector",     # Search strategy for relations
    document_search="vector"      # Search strategy for documents
)
```

## Example Usage

```python
import requests
import json

url = "http://localhost:8000/search"
payload = {
    "global_query": "What are the main components of a nuclear reactor?"
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

## Development

### Project Structure

```
Knowledge_Graph_Retriever_API/
├── api.py                   # FastAPI application
├── api_test.ipynb           # Example usage notebook
├── prompt_templates/        # Prompt templates for LLMs
├── retriever/              # Retriever implementations
│   ├── BM25_retriever.py
│   ├── Base_Retriever.py
│   ├── Connecting_Dots.py
│   ├── HyDE_retriever.py
│   └── vector_search.py
├── traversing_process.py    # Core search logic
└── utils/                   # Utility functions
```

