#!/usr/bin/env python3
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        """
        self.X = X_init  # Sampled inputs
        self.Y = Y_init  # Corresponding outputs
        self.l = l  # Length scale for the RBF kernel
        self.sigma_f = sigma_f  # Standard deviation for the function
        # Calculate the covariance matrix (kernel matrix)
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
Calculates the covariance kernel matrix
        """
    # Squared distance between points
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
     # RBF kernel formula
        return self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)
