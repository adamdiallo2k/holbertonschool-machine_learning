#!/usr/bin/env python3
""" Initializes cluster centroids for K-means """

import numpy as np


def initialize(X, k):
    """
    Initializes centroids for K-means clustering.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be used
        k: positive integer containing the number of clusters

    Returns:
        centroids: numpy.ndarray of shape (k, d) containing the initialized centroids,
                   or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if k <= 0 or len(X.shape) != 2:
        return None

    n, d = X.shape

    # Compute the minimum and maximum values for each dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids using uniform distribution between min_vals and max_vals
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
