from embeddings.embedder import Embedder

texts = [
    "Space exploration is fascinating",
    "The baseball game was exciting"
]

embedder = Embedder()

vectors = embedder.encode(texts)

print("Vector shape:", vectors.shape)
print("First vector preview:", vectors[0][:10])