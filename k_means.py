import numpy as np


class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        for i in range(self.max_iterations):
            self.centroids = X[np.random.choice(range(len(X)), self.k, replace=False)]

            cluster_assignments = []

            for idx in range(len(X)):
                dist = np.linalg.norm(X[idx] - self.centroids, axis=1)
                cluster_assignments.append(np.argmin(dist))

            for k in range(self.k):
                cluster_datapoints = X[np.where(np.array(cluster_assignments) == k)]
                if len(cluster_assignments):
                    self.centroids[k] = np.mean(cluster_datapoints, axis=0)

            if i > 0 and np.array_equal(self.centroids, previous_centroids):
                break

            previous_centroids = self.centroids

        self.cluster_assignment = cluster_assignments

    def predict(self, x):
        res = []
        for idx in range(len(x)):
            dist = np.linalg.norm(x[idx] - self.centroids, axis=-1)
            assignment = np.argmin(dist)
            res.append(assignment)
        return res


x1 = np.random.randn(5, 2) + 5
x2 = np.random.randn(5, 2) - 5
X = np.concatenate([x1, x2], axis=0)

# Initialize the KMeans object with k=3
kmeans = KMeans(k=2)

# Fit the k-means model to the dataset
kmeans.fit(X)

# Get the cluster assignments for the input dataset
cluster_assignments = kmeans.predict(X)

# Print the cluster assignments
print(cluster_assignments)

# Print the learned centroids
print(kmeans.centroids)
