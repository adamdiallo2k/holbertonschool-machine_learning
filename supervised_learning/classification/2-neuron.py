#!/usr/bin/env python3
"""commented module """
import numpy as np


import numpy as np

class Neuron:
    """ Represents a single neuron performing binary classification. """

    def __init__(self, nx):
        """
        Initialize a neuron with the given attributes.

        Parameters:
        nx (int): Number of input features for the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Weights, initialized randomly
        self.__b = 0  # Bias, initialized to 0
        self.__A = 0  # Activated output, initialized to 0

    @property
    def W(self):
        """ Getter for W. """
        return self.__W

    @property
    def b(self):
        """ Getter for b. """
        return self.__b

    @property
    def A(self):
        """ Getter for A. """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): numpy.ndarray with shape (nx, m) containing the input data.

        Updates the private attribute __A.
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear combination of inputs and weights, plus bias
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A
