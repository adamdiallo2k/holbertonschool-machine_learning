#!/usr/bin/env python3
"""
This script initializes centroids for K-means clustering using a uniform 
distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes centroids for K-means using a uniform distribution.
    
    Args:
    - X (numpy.ndarray): Dataset (n, d) where n is the number of points, 
      and d is the number of dimensions.
    - k (int): Number of clusters.
    
    Returns:
    - numpy.ndarray: Initialized centroids of shape (k, d), or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
       not isinstance(k, int) or k <= 0:
        return None
    
    # Get minimum and maximum values for each dimension
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    
    # Generate centroids using a uniform distribution
    centroids = np.random.uniform(low=min_values, high=max_values, size=(k, X.shape[1]))
    
    return centroids
