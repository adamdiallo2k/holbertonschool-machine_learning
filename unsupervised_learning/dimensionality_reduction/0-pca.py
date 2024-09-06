#!/usr/bin/env python3
"""
    Perform Principal Component Analysis (PCA) on a dataset.
"""
import numpy as np

def pca(X, var=0.95):
    """
    Perform PCA on the dataset X.

    Args:
    X : numpy.ndarray of shape (n, d)
        - n is the number of data points
        - d is the number of dimensions for each data point
        The dataset should already be centered (mean = 0 for each feature).

    var : float, optional (default=0.95)
        The fraction of the variance to preserve after the PCA transformation.

    Returns:
    W : numpy.ndarray of shape (d, k)
        The matrix containing the principal components that explain the given
        variance. The dataset X can be projected onto this matrix for
        dimensionality reduction.
    """
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute the explained variance for each singular value
    explained_variance = (S ** 2) / np.sum(S ** 2)
    
    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    # Find the minimum number of components required to retain the target variance
    k = np.searchsorted(cumulative_variance, var) + 1

    # Extract the first k components (columns) from Vt, which correspond to the principal components
    W = Vt[:k].T

    return W
