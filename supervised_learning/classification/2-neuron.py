#!/usr/bin/env python3
import numpy as np
"""commented module"""


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
        Calculates the forward
         propagation of the neuron.

        Parameters:
        X (numpy.ndarray): numpy.ndarray with shape
         (nx, m) containing the input data.

        Returns:
        numpy.ndarray: The activated
        output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model
        using logistic regression.

        Parameters:
        Y (numpy.ndarray): numpy.ndarray with shape
        (1, m) containing the correct labels.
        A (numpy.ndarray): numpy.ndarray
         with shape (1, m) containing the activated output.

        Returns:
        numpy.ndarray:
        The cost of the model.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A)+ (1 - Y) * 
        np.log(1.0000001 - A))
        return cost
