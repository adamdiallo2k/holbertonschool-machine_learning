#!/usr/bin/env python3
import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a uniform distribution.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
       not isinstance(k, int) or k <= 0:
        return None
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    centroids = np.random.uniform(low=min_values, high=max_values, size=(k, X.shape[1]))
    return centroids

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
       not isinstance(k, int) or k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)  # Initialize centroids
    if C is None:
        return None, None

    clss = np.zeros(n)
    prev_C = np.copy(C)

    for i in range(iterations):  # First loop: iterate over the allowed number of iterations
        # Step 1: Assign points to nearest centroid (vectorized, no loop here)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Step 2: Update centroids (second loop: iterate over each centroid)
        for j in range(k):
            if np.any(clss == j):
                C[j] = np.mean(X[clss == j], axis=0)
            else:
                # Reinitialize the centroid if it has no points
                C[j] = np.random.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0))

        # Check for convergence (if centroids don't change)
        if np.all(C == prev_C):
            break

        prev_C = np.copy(C)

    return C, clss
