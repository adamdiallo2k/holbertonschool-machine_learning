import numpy as np
from sklearn.preprocessing import StandardScaler

def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on the dataset X.

    Args:
    X : numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points and d is the number of dimensions.
    var : float
        The fraction of the variance that the PCA transformation should maintain.

    Returns:
    W : numpy.ndarray of shape (d, nd)
        The weights matrix that maintains var fraction of X's original variance.
        nd is the new dimensionality of the transformed X.
    """
    # Optional Step: Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Step 1: Compute the covariance matrix of the standardized dataset X
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # Step 2: Perform Eigen Decomposition on the covariance matrix using eigh
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 3: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 4: Calculate the cumulative variance
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    print("Cumulative Variance:", cumulative_variance)  # Debug statement
    
    # Step 5: Determine the number of components to maintain the desired variance
    nd = np.argmax(cumulative_variance >= var) + 1
    print("Number of components selected (nd):", nd)  # Debug statement
    
    # Step 6: Return the weight matrix W
    W = sorted_eigenvectors[:, :nd]
    
    return W
