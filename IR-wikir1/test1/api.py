from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from test1.retrieving_functions import get_top_related_docs, get_top_related_docs_clustered

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow requests from specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Document(BaseModel):
    doc_id: str
    text: str


class QueryResponse(BaseModel):
    top_docs: List[str]
    top_docs_text: List[str]
    top_scores: List[float]


class QueryRequest(BaseModel):
    query: str
    num_docs: int = 10


@app.get('/api/get_top_related_docs', response_model=QueryResponse)
def api_get_top_related_docs(query: str, num_docs: int = 10):
    top_docs, top_docs_text, top_scores = get_top_related_docs(query, num_docs)

    response = QueryResponse(
        top_docs=top_docs,
        top_docs_text=top_docs_text,
        top_scores=top_scores
    )
    return response


@app.get('/api/get_top_related_docs_cluster', response_model=QueryResponse)
def get_top_related_docs_cluster(query: str, num_docs: int = 10):
    top_docs, top_docs_text, top_scores = get_top_related_docs_clustered(query, num_docs)

    response = QueryResponse(
        top_docs=top_docs,
        top_docs_text=top_docs_text,
        top_scores=top_scores
    )
    return response


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=9001)
