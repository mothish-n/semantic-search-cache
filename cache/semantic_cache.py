import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.60):

        self.similarity_threshold = similarity_threshold

        # cache grouped by clusters
        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0


    def search_cache(self, query_embedding, cluster):

        # if cluster has no cached queries
        if cluster not in self.cache:

            self.miss_count += 1
            return None

        best_match = None
        best_score = 0

        for entry in self.cache[cluster]:

            score = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.similarity_threshold:

            self.hit_count += 1

            return {
                "cache_hit": True,
                "matched_query": best_match["query"],
                "similarity_score": float(best_score),
                "result": best_match["result"],
                "cluster": cluster
            }

        self.miss_count += 1
        return None


    def add_to_cache(self, query, embedding, result, cluster):

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append({
            "query": query,
            "embedding": embedding,
            "result": result
        })


    def cache_stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        total_requests = self.hit_count + self.miss_count

        hit_rate = 0

        if total_requests > 0:
            hit_rate = self.hit_count / total_requests

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 3)
        }


    def clear_cache(self):

        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0