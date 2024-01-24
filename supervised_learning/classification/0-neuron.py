#!/usr/bin/env python3
import numpy as np
"""commented module """
class Neuron:
    def __init__(self, nx):
        """
        Constructor for the Neuron class.

        Parameters:
        nx (int): Number of input features to the neuron.

        Attributes:
        W (numpy.ndarray): The weights vector for the neuron.
        b (float): The bias for the neuron.
        A (float): The activated output of the neuron (prediction).
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize the weights vector with a random normal distribution
        self.W = np.random.randn(nx)

        # Initialize the bias to 0
        self.b = 0

        # Initialize the activated output to 0
        self.A = 0
