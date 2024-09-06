import numpy as np

def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on the dataset X.
    
    Args:
    X : numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points and d is the number of dimensions.
    var : float
        The fraction of the variance that the PCA transformation should maintain.
    
    Returns:
    X_pca : numpy.ndarray of shape (n, nd)
        The dataset X transformed to the new lower-dimensional space.
    W : numpy.ndarray of shape (d, nd)
        The weights matrix that maintains var fraction of X's original variance.
        nd is the new dimensionality of the transformed X.
    """
    # Step 1: Standardize the data by subtracting the mean and dividing by the standard deviation
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_centered = (X - X_mean) / X_std
    
    # Step 2: Compute the covariance matrix of the standardized dataset
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Step 3: Perform Eigen Decomposition on the covariance matrix using eigh (for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Calculate the cumulative variance explained by each principal component
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    # Step 6: Determine the number of components that maintain the desired variance
    nd = np.argmax(cumulative_variance >= var) + 1
    
    # Step 7: Select the eigenvectors (principal components) corresponding to nd components
    W = sorted_eigenvectors[:, :nd]
    
    # Step 8: Project the data onto the new principal component space
    X_pca = np.dot(X_centered, W)
    
    # Return the projected data and the weight matrix
    return X_pca, W
