#!/usr/bin/env python3
 """
    Calculates the total intra-cluster variance for a dataset.
"""


import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    :param X: ndarray of shape (n, d) containing the dataset
    :param C: ndarray of shape (k, d) containing the centroid means for each cluster

    :return: var, the total variance, or None on failure
    """
    # Validate input
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Step 1: Calculate distances to centroids
    # X[:, np.newaxis] makes X (n, 1, d) and C (1, k, d), so the subtraction broadcasts
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    
    # Step 2: Find the closest centroid for each point
    closest_centroid_idx = np.argmin(distances, axis=1)
    
    # Step 3: Calculate the total intra-cluster variance
    variance = np.sum(np.linalg.norm(X - C[closest_centroid_idx], axis=1) ** 2)
    
    return variance
