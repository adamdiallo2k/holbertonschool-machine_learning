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
        # Calculate the covariance matrix (kernel matrix)
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using the RBF kernel.

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        # Calculate the squared distance between each pair of points
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        # Apply the RBF kernel formula
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
