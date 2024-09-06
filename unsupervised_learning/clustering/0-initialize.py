#!/usr/bin/env python3
import numpy as np

def initialize(X, k):
    """
    Initializes centroids for K-means clustering.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters

    Returns:
        numpy.ndarray: Initialized centroids, or None on failure
    """
    # Check for valid inputs
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if k <= 0 or len(X.shape) != 2:
        return None

    # Get the min and max values for each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Generate k centroids within the bounds of the dataset
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))

    return centroids
