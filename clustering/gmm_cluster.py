import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class FuzzyCluster:

    def __init__(self, n_clusters=15, pca_dim=50):

        self.n_clusters = n_clusters

        # reduce embedding dimension
        self.pca = PCA(n_components=pca_dim)

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-4,
            max_iter=300,
            random_state=42
        )

    def fit(self, embeddings):

        embeddings = np.array(embeddings)

        reduced = self.pca.fit_transform(embeddings)

        self.model.fit(reduced)

    def get_cluster_distribution(self, embeddings):

        embeddings = np.array(embeddings)

        reduced = self.pca.transform(embeddings)

        probs = self.model.predict_proba(reduced)

        return probs