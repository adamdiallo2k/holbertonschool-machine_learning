#!/usr/bin/env python3
import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    X: numpy.ndarray of shape (n, d) containing the dataset.
    k: positive integer containing the number of clusters.
    
    Returns: numpy.ndarray of shape (k, d) containing the initialized centroids.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    centroids = np.random.uniform(min_values, max_values, (k, d))
    
    return centroids
