from data.dataset_loader import load_newsgroups
from data.text_cleaner import clean_text
from embeddings.embedder import Embedder
from vector_store.faiss_store import FAISSStore

dataset_path = r"C:\Users\mothi\OneDrive\Desktop\semantic-search-cache\data\20_newsgroups"

docs, labels = load_newsgroups(dataset_path)

docs = docs[:1000]  # use smaller subset for testing

clean_docs = [clean_text(doc) for doc in docs]

embedder = Embedder()

embeddings = embedder.encode(clean_docs)

store = FAISSStore(embeddings.shape[1])

store.add_documents(embeddings, clean_docs)

query = "space shuttle mission"

query_vector = embedder.encode([query])[0]

results, distances = store.search(query_vector)

print("Query:", query)
print("\nTop result preview:\n")
print(results[0][:500])