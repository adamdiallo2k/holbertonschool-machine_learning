#!/usr/bin/env python3
"""
    Clustering using K-means algorithm
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
        Performs K-means clustering on a dataset.

    :param X: ndarray of shape (n, d) containing the dataset
        n: number of data points
        d: number of dimensions for each data point
    :param k: int, number of clusters
    :param iterations: positive int, maximum number of iterations

    :return: C, clss or None, None on failure
        C: ndarray of shape (k, d) containing the centroid means for each cluster
        clss: ndarray of shape (n,) containing the index of the cluster in C
              that each data point belongs to
    """
    # Validate inputs
    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids with multivariate uniform distribution
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    new_centroids = np.empty((k, d), dtype=X.dtype)

    # Main loop for K-means
    for i in range(iterations):
        # Step 1: Compute distances between data points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Step 2: Update centroids
        for j in range(k):
            mask = (clss == j)
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                # Reinitialize centroid if no points are assigned to it
                new_centroids[j] = np.random.uniform(low=low, high=high, size=(1, d))

        # Step 3: Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids.copy()

    return centroids, clss
