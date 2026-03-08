from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache

app = FastAPI()

# Initialize components
embedder = Embedder()
cache = SemanticCache(similarity_threshold=0.60)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Semantic Search API running"}


@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query

    # Generate embedding
    embedding = embedder.encode([query])[0]

    # Temporary cluster id
    cluster_id = 0

    # Search semantic cache
    cache_result = cache.search_cache(embedding, cluster_id)

    if cache_result:
        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["result"],
            "dominant_cluster": cluster_id
        }

    # Simulated result
    result = f"Search results for: {query}"

    # Add to cache
    cache.add_to_cache(
        query=query,
        embedding=embedding,
        result=result,
        cluster=cluster_id
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.cache_stats()


@app.delete("/cache")
def clear_cache():
    cache.clear_cache()
    return {"message": "Cache cleared"}