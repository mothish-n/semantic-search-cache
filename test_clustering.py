from data.dataset_loader import load_newsgroups
from data.text_cleaner import clean_text
from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster

# dataset location
dataset_path = r"C:\Users\mothi\OneDrive\Desktop\semantic-search-cache\data\20_newsgroups"

# -----------------------------
# Load dataset
# -----------------------------
docs, labels = load_newsgroups(dataset_path)

# use subset for faster testing
docs = docs[:1000]

# -----------------------------
# Clean documents
# -----------------------------
clean_docs = [clean_text(doc) for doc in docs]

# -----------------------------
# Generate embeddings
# -----------------------------
embedder = Embedder()

print("\nGenerating embeddings...\n")

embeddings = embedder.encode(clean_docs)

# -----------------------------
# Train fuzzy clustering model
# -----------------------------
clusterer = FuzzyCluster(n_clusters=15)

print("\nTraining clustering model...\n")

clusterer.fit(embeddings)

# -----------------------------
# Get cluster probabilities
# -----------------------------
probs = clusterer.get_cluster_distribution(embeddings)

# -----------------------------
# Display cluster distributions
# -----------------------------
print("\nCluster distributions for first 5 documents:\n")

for i in range(5):

    print(f"\nDocument {i}")

    dist = probs[i]

    for cluster_id, p in enumerate(dist):

        if p > 0.05:
            print(f"Cluster {cluster_id} → {p:.3f}")