#!/usr/bin/env python3
"""commented module """


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

        self.__W = [0] * nx  # Weights, initialized to zeroes
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
        X (list of lists): List of lists with shape (nx, m) containing the input data.
        """
        m = len(X[0])
        Z = [self.__b] * m
        for i in range(len(self.__W)):
            for j in range(m):
                Z[j] += self.__W[i] * X[i][j]

        self.__A = [1 / (1 + self.exp(-z)) for z in Z]
        return self.__A

    def exp(self, x):
        """
        Calculates the exponential of x.
        """
        n = 10  # Precision of the approximation
        return sum([x**i / self.factorial(i) for i in range(n)])

    def factorial(self, n):
        """
        Calculates the factorial of n.
        """
        return 1 if n == 0 else n * self.factorial(n - 1)

    def log(self, x):
        """
        Calculates the natural logarithm of x.
        """
        n = 100  # Precision of the approximation
        if x - 1 < 0:
            return -sum([((1 - x) ** i) / i for i in range(1, n)])
        else:
            return sum([((x - 1) ** i) / i for i in range(1, n)])

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (list): List containing the correct labels.
        A (list): List containing the activated output.

        Returns:
        The cost.
        """
        m = len(Y)
        cost = - sum([Y[i] * self.log(A[i]) + (1 - Y[i]) * self.log(1.0000001 - A[i]) for i in range(m)]) / m
        return cost
