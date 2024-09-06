#!/usr/bin/env python3
"""
This script contains a function `initialize` that initializes centroids for K-means clustering.
The centroids are initialized using a multivariate uniform distribution based on the provided dataset.
The function takes a dataset `X` and the number of clusters `k`, and returns the initialized centroids.

The initialization ensures that the centroids are selected randomly but within the range of the data points
for each dimension, ensuring they are well spread out in the dataset.
"""

import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a uniform distribution.
    
    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d), where n is the number of data points and d is the number of dimensions.
    - k (int): Number of clusters.
    
    Returns:
    - centroids (numpy.ndarray): Initialized centroids of shape (k, d), or None on failure.
    """
    # Validate inputs: X should be a 2D numpy array and k should be a positive integer
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or not isinstance(k, int) or k <= 0:
        return None
    
    # Extract the number of data points (n) and dimensions (d) from X
    n, d = X.shape
    
    # Compute the minimum and maximum values along each dimension of the dataset
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    
    # Generate k centroids, each with d dimensions, from the uniform distribution
    centroids = np.random.uniform(low=min_values, high=max_values, size=(k, d))
    
    return centroids
