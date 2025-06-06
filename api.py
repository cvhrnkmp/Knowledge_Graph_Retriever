from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from traversing_process import DRIFTSearch
from langchain.schema import Document

app = FastAPI()
searcher = DRIFTSearch(
    chat_model="qwen3:14b",
    embeddings_model="jina/jina-embeddings-v2-base-de",
    dataset="Pre_Framatome"
    )
# Initialize search strategies
searcher.init_search(
    "vector",
    "vector",
    "vector",
    "vector"
)
class SearchRequest(BaseModel):
    global_query: str
    


@app.post("/search", response_model=SearchRequest)
async def search(req: SearchRequest):
    try:
        result = await searcher.search(req.global_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result
