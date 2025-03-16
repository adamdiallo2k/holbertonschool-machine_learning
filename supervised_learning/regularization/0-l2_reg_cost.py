#!/usr/bin/env python3
"""Module commenté pour calculer le coût d'un réseau avec régularisation L2."""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calcule le coût avec régularisation L2 d'un réseau de neurones.

    Arguments :
    cost -- Coût sans régularisation L2
    lambtha -- Paramètre de régularisation
    weights -- Dictionnaire de poids et biais du réseau
    L -- Nombre de couches dans le réseau
    m -- Nombre de données utilisées

    Returns :
    Coût total avec régularisation L2.
    """

    l2_penalty = sum([np.sum(np.square(weights['W' + str(layer)]))
                      for layer in range(1, L + 1)])

    cost_l2 = cost + (lambtha / (2 * m)) * l2_penalty
    return cost_l2
