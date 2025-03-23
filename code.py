import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Define the correct file path
directory = "C:/Users/KIIT/PycharmProjects/PythonProject/MLASS3"
file_name = "kmeans_blobs.csv"
file_path = os.path.join(directory, file_name)

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(df)


def initialize_centroids(data, k):
    """Randomly initialize k centroids from the data."""
    np.random.seed(42)
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, labels, k):
    """Recalculate centroids as the mean of assigned points."""
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])


def kmeans(data, k, max_iters=100, tol=1e-4):
    """K-means clustering algorithm."""
    centroids = initialize_centroids(data, k)

    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids


def plot_clusters(data, labels, centroids, k):
    """Plot the clustered data."""
    plt.figure(figsize=(6, 5))
    for i in range(k):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel("x1 (normalized)")
    plt.ylabel("x2 (normalized)")
    plt.title(f"K-Means Clustering (k={k})")
    plt.legend()
    plt.show()


# Run K-means for k=2 and k=3
labels_k2, centroids_k2 = kmeans(data, k=2)
labels_k3, centroids_k3 = kmeans(data, k=3)

# Convert labels to numpy array for easier indexing
labels_k2 = np.array(labels_k2)
labels_k3 = np.array(labels_k3)

# Plot for k=2
plot_clusters(data, labels_k2, centroids_k2, k=2)

# Plot for k=3
plot_clusters(data, labels_k3, centroids_k3, k=3)
