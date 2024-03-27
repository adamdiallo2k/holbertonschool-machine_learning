#!/usr/bin/env python3
"""Commented module """


import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):


    """
    Calcule le coût d'un réseau de neurones avec régularisation L2.

    Arguments:
    cost -- le coût du réseau sans régularisation L2
    lambtha -- le paramètre de régularisation
    weights -- un dictionnaire des poids et biais (numpy.ndarrays) du réseau de neurones
    L -- le nombre de couches dans le réseau de neurones
    m -- le nombre de points de données utilisés

    Returns:
    Le coût du réseau en prenant en compte la régularisation L2.
    """

    l2_penalty = 0
    for l in range(1, L + 1):
        # La clé pour les poids de chaque couche suit le format 'Wl', où l est le numéro de la couche
        Wl = weights['W' + str(l)]
        # Additionne la norme de Frobenius des poids (somme des carrés des éléments) pour chaque couche
        l2_penalty += np.sum(np.square(Wl))

    # Calcul du coût avec régularisation L2
    cost_l2 = cost + (lambtha / (2 * m)) * l2_penalty
    return cost_l2
