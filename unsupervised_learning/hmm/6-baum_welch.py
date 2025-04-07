#!/usr/bin/env python3
"""
Performs the Baum-Welch algorithm for a Hidden Markov Model (HMM).

Observations: shape (T,), the indices of observed symbols
Transition: shape (M, M), transition probabilities between hidden states
Emission: shape (M, N), emission probabilities of observations given a hidden state
Initial: shape (M, 1), initial state probabilities
iterations: number of EM iterations to perform

Returns: (Transition, Emission) after the algorithm converges or
         (None, None) on any failure (e.g., invalid inputs).
"""


import numpy as np


def forward(Obs, Transition, Emission, Initial):
    """
    Performs the forward algorithm:
    Obs is a numpy.ndarray of shape (T,) that contains the index of each observation
    Transition is shape (M, M)
    Emission is shape (M, N)
    Initial is shape (M, 1)
    Returns: alpha, a numpy.ndarray of shape (M, T) containing the forward path probabilities
    """
    T = Obs.shape[0]
    M = Transition.shape[0]

    alpha = np.zeros((M, T))

    # Initialize alpha for t = 0
    alpha[:, 0] = Initial.T * Emission[:, Obs[0]]

    # Recursively fill in alpha
    for t in range(1, T):
        for j in range(M):
            alpha[j, t] = np.sum(alpha[:, t - 1] * Transition[:, j]) * Emission[j, Obs[t]]

    return alpha


def backward(Obs, Transition, Emission, Initial):
    """
    Performs the backward algorithm:
    Obs is a numpy.ndarray of shape (T,) that contains the index of each observation
    Transition is shape (M, M)
    Emission is shape (M, N)
    Initial is shape (M, 1)
    Returns: beta, a numpy.ndarray of shape (M, T) containing the backward path probabilities
    """
    T = Obs.shape[0]
    M = Transition.shape[0]

    beta = np.zeros((M, T))

    # Initialize beta for t = T-1
    beta[:, T - 1] = 1

    # Recursively fill in beta
    for t in range(T - 2, -1, -1):
        for i in range(M):
            beta[i, t] = np.sum(
                Transition[i, :] * Emission[:, Obs[t + 1]] * beta[:, t + 1]
            )

    return beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model (HMM).

    Parameters:
    - Observations: numpy.ndarray of shape (T,) index of each observation
    - Transition: numpy.ndarray of shape (M, M)
    - Emission: numpy.ndarray of shape (M, N)
    - Initial: numpy.ndarray of shape (M, 1)
    - iterations: number of EM updates to perform

    Returns:
    - Transition, Emission (updated) or (None, None) on failure
    """

    # Basic checks
    if (type(Observations) is not np.ndarray or len(Observations.shape) != 1):
        return None, None
    if (type(Transition) is not np.ndarray or len(Transition.shape) != 2):
        return None, None
    if (type(Emission) is not np.ndarray or len(Emission.shape) != 2):
        return None, None
    if (type(Initial) is not np.ndarray or len(Initial.shape) != 2):
        return None, None

    T = Observations.shape[0]
    M, M2 = Transition.shape
    if M != M2:
        return None, None

    M2, N = Emission.shape
    if M != M2:
        return None, None

    if Initial.shape[0] != M or Initial.shape[1] != 1:
        return None, None

    # Copy to avoid modifying original arrays
    Transition = Transition.copy()
    Emission = Emission.copy()
    Initial = Initial.copy()

    for _ in range(iterations):
        # E-Step: calculate forward and backward probabilities
        alpha = forward(Observations, Transition, Emission, Initial)
        beta = backward(Observations, Transition, Emission, Initial)

        # Calculate xi and gamma
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(alpha[:, t], beta[:, t])
            denominator = np.sum(denominator * Transition)
            for i in range(M):
                numerator = alpha[i, t] * Transition[i, :] * Emission[:, Observations[t + 1]] * beta[:, t + 1]
                xi[i, :, t] = numerator / np.sum(numerator) if np.sum(numerator) != 0 else 0

        gamma = np.sum(xi, axis=1)  # shape (M, T - 1)

        # For the last time step, we also need gamma(T-1, i)
        last_gamma = (alpha[:, T - 1] * beta[:, T - 1]) / np.sum(alpha[:, T - 1] * beta[:, T - 1])
        gamma = np.hstack((gamma, last_gamma.reshape(M, 1)))

        # M-Step: update Transition
        for i in range(M):
            denom = np.sum(gamma[i, :-1])
            for j in range(M):
                # Sum xi over t from 0 to T-2
                Transition[i, j] = np.sum(xi[i, j, :]) / denom if denom != 0 else 0

        # Update Emission
        for k in range(N):
            mask = (Observations == k)
            for i in range(M):
                numerator = np.sum(gamma[i, mask])
                denominator = np.sum(gamma[i, :])
                Emission[i, k] = numerator / denominator if denominator != 0 else 0

    return Transition, Emission
