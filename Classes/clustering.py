import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class clustering:

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        self.cluster_labels = None

    def fit_predict(self, X):
        """
        Cluster pixels using their feature matrix X.

        X: array of shape (number of pixels, number of features)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.cluster_labels = self.model.fit_predict(X_scaled)
        return self.cluster_labels

    def reshape_clusters(self, image_shape):
        """
        Reshape the cluster labels into the original image dimensions.
        """
        if self.cluster_labels is None:
            raise ValueError("Run fit_predict() first.")

        return self.cluster_labels.reshape(image_shape)