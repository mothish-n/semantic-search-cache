import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class FuzzyCluster:

    def __init__(self, n_clusters=15, random_state=42):

        self.n_clusters = n_clusters

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    def fit(self, embeddings):

        embeddings = normalize(np.array(embeddings))

        self.kmeans.fit(embeddings)

        self.centers = self.kmeans.cluster_centers_

    def get_cluster_distribution(self, embeddings):

        embeddings = normalize(np.array(embeddings))

        similarity = np.dot(embeddings, self.centers.T)

        # convert similarity to probability
        similarity = similarity - similarity.min(axis=1, keepdims=True)

        probs = similarity / similarity.sum(axis=1, keepdims=True)

        return probs