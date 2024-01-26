#!/usr/bin/env python3
import numpy as np

class Neuron:
    """ Represents a single neuron performing binary classification. """

    def __init__(self, nx):
        """
        Initialize a neuron with the given attributes.

        Parameters:
        nx (int): Number of input features for the neuron.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)  # Weights, initialized randomly
        self.__b = 0  # Bias, initialized to 0
        self.__A = 0  # Activated output, initialized to 0

    @property
    def W(self):
        """ Getter for the weight. """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias. """
        return self.__b

    @property
    def A(self):
        """ Getter for the activated output. """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using a sigmoid activation function.

        Parameters:
        X (numpy.ndarray): numpy.ndarray with shape (nx, m) containing the input data.

        Returns:
        numpy.ndarray: The activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b  # Compute the linear combination
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A
