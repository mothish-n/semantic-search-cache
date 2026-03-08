from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache

embedder = Embedder()

cache = SemanticCache(similarity_threshold=0.60)

query1 = "What is space exploration?"

query_embedding1 = embedder.encode([query1])[0]

cache.add_to_cache(
    query=query1,
    embedding=query_embedding1,
    result="Information about space exploration",
    cluster=3
)

query2 = "Tell me about space missions"

query_embedding2 = embedder.encode([query2])[0]


from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(
    [query_embedding1],
    [query_embedding2]
)[0][0]

print("\nActual similarity between queries:", sim)

result = cache.search_cache(query_embedding2)

print("\nCache search result:\n")
print(result)

print("\nCache statistics:\n")
print(cache.cache_stats())