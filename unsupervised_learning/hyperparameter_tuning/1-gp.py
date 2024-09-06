#!/usr/bin/env python3
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""
    
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        
        Args:
            X_init: numpy.ndarray of shape (t, 1) - inputs already sampled
            Y_init: numpy.ndarray of shape (t, 1) - outputs of the black-box function
            l: length parameter for the kernel
            sigma_f: standard deviation of the output
        """
        self.X = X_init  # Sampled inputs
        self.Y = Y_init  # Corresponding outputs
        self.l = l  # Length scale for the RBF kernel
        self.sigma_f = sigma_f  # Standard deviation for the function
        self.K = self.kernel(X_init, X_init)  # Covariance matrix of training points
    
    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using the RBF kernel.
        
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
            
        Returns:
            Covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)
    
    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian process.
        
        Args:
            X_s: numpy.ndarray of shape (s, 1) containing all of the points whose mean and variance should be calculated
            s: number of sample points
        
        Returns:
            mu: numpy.ndarray of shape (s,) containing the mean for each point in X_s, respectively
            sigma: numpy.ndarray of shape (s,) containing the variance for each point in X_s, respectively
        """
        # Covariance between training data (X) and new points (X_s)
        K_s = self.kernel(self.X, X_s)
        # Covariance between the new points
        K_ss = self.kernel(X_s, X_s)
        # Inverse of the covariance matrix of the training data
        K_inv = np.linalg.inv(self.K)
        
        # Compute the mean
        mu_s = K_s.T @ K_inv @ self.Y
        mu = mu_s.flatten()  # Convert to 1D array

        # Compute the covariance (variance)
        cov_s = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov_s)  # Extract the diagonal (variances)

        return mu, sigma
