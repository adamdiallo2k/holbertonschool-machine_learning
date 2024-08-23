#!/usr/bin/env python3
import numpy as np

def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on the dataset X and reduces its dimensionality.

    Args:
    X : numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points and d is the number of dimensions.
    ndim : int
        The new dimensionality of the transformed X.

    Returns:
    T : numpy.ndarray of shape (n, ndim)
        The transformed version of X with reduced dimensionality.
    """
    # Step 1: Compute the covariance matrix of the dataset X
    cov_matrix = np.cov(X, rowvar=False)
    
    # Step 2: Perform Eigen Decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 3: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 4: Select the top 'ndim' eigenvectors
    W = sorted_eigenvectors[:, :ndim]
    
    # Step 5: Transform the dataset X to the new lower-dimensional space
    T = np.dot(X, W)
    
    return T
