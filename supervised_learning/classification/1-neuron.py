#!/usr/bin/env python3
"""commented module """
import numpy as np


class Neuron:
    """ Représente un seul neurone effectuant une classification binaire. """

    def __init__(self, nx):
        """
        Initialise un neurone avec les attributs donnés.

        Parameters:
        nx (int): Nombre de caractéristiques d'entrée du neurone.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Poids, initialisés aléatoirement
        self.__b = 0  # Biais, initialisé à 0
        self.__A = 0  # Sortie activée, initialisée à 0

    @property
    def W(self):
        """ Getter pour W. """
        return self.__W

    @property
    def b(self):
        """ Getter pour b. """
        return self.__b

    @property
    def A(self):
        """ Getter pour A. """
        return self.__A
    
    def forward_prop(self, X):
        
