# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a lightweight semantic search system built on the **20 Newsgroups dataset**. The system demonstrates how modern AI retrieval pipelines combine **vector embeddings, clustering, semantic caching, and API services** to efficiently process natural language queries.

The architecture includes:

* Sentence embeddings for semantic understanding
* Vector similarity search
* Fuzzy clustering of documents
* A semantic cache that detects paraphrased queries
* A FastAPI service exposing the system through REST endpoints

The project was developed as part of the **Trademarkia AI/ML Engineer Assignment**.

---

# Dataset

The system uses the **20 Newsgroups dataset**, containing approximately **20,000 documents** across **20 topic categories**.

Example categories include:

* comp.graphics
* sci.space
* rec.sport.baseball
* talk.politics.guns
* soc.religion.christian

Dataset source:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

---

# System Architecture

```
Dataset
   ↓
Text Preprocessing
   ↓
Sentence Embeddings (MiniLM)
   ↓
Vector Database (FAISS)
   ↓
Fuzzy Clustering
   ↓
Semantic Cache
   ↓
FastAPI Service
```

---

# Part 1 — Embedding and Vector Database

Documents are converted into dense vector embeddings using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Reasons for choosing this model:

* Lightweight and fast
* Strong semantic performance
* 384-dimensional embeddings suitable for retrieval systems

The embeddings are indexed using **FAISS**, which enables efficient nearest-neighbor search in vector space.

---

# Part 2 — Fuzzy Clustering

The 20 Newsgroups dataset contains overlapping topics, meaning documents may belong to multiple semantic groups.

Instead of hard clustering, this system produces **soft cluster memberships**.

Each document receives a **probability distribution over clusters**:

Example:

```
Document A
Cluster 1 → 0.16
Cluster 4 → 0.13
Cluster 12 → 0.14
```

This approach better represents the ambiguous nature of real-world text data.

Clustering was implemented using:

```
KMeans + similarity-based soft assignment
```

Cluster probabilities are derived from cosine similarity between embeddings and cluster centroids.

---

# Part 3 — Semantic Cache

Traditional caching fails when users phrase the same query differently.

Example:

```
"What is space exploration?"
"Tell me about space missions"
```

These queries are semantically similar but textually different.

The semantic cache solves this by:

1. Embedding incoming queries
2. Comparing them to cached query embeddings
3. Returning cached results if similarity exceeds a threshold

Cosine similarity is used to measure query similarity.

### Cache Threshold Analysis

| Threshold | Behavior                          |
| --------- | --------------------------------- |
| 0.85      | Only near-identical queries match |
| 0.70      | Captures paraphrases              |
| 0.60      | Balanced recall and precision     |

Final chosen threshold:

```
0.60
```

This value captures semantic similarity while minimizing incorrect cache hits.

---
## Setup

pip install -r requirements.txt
uvicorn api.main:app --reload



# Part 4 — FastAPI Service

The system is exposed through a REST API using **FastAPI**.

### POST /query

Accepts a natural language query.

Example request:

```json
{
  "query": "Tell me about space missions"
}
```

Example response:

```json
{
  "query": "Tell me about space missions",
  "cache_hit": true,
  "matched_query": "What is space exploration?",
  "similarity_score": 0.76,
  "result": "Search results for: What is space exploration?",
  "dominant_cluster": 0
}
```

---

### GET /cache/stats

Returns cache statistics.

Example:

```json
{
  "total_entries": 1,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
```

---

### DELETE /cache

Clears the semantic cache.

---

# Running the Project

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Start the API

```
uvicorn api.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

# Project Structure

```
semantic-search-cache
│
├── api
│   └── main.py
│
├── cache
│   └── semantic_cache.py
│
├── clustering
│   └── fuzzy_cluster.py
│
├── embeddings
│   └── embedder.py
│
├── vector_store
│   └── faiss_store.py
│
├── data
│
├── test_cache.py
├── test_clustering.py
├── test_embedding.py
├── test_loader.py
│
├── requirements.txt
└── README.md
```

---

# Future Improvements

Potential improvements include:

* Cluster-aware cache indexing
* Persistent vector storage
* Distributed caching
* Query result ranking
* Docker containerization for deployment

---

# Author

AI/ML Engineer Assignment Submission

Candidate: [N Mothish]

GitHub Repository: [(https://github.com/mothish-n/semantic-search-cache)]
