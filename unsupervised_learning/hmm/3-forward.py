#!/usr/bin/env python3
"""
    HMM : Hidden Markov Models
        The Forward Algorithm
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
        performs the forward algorithm for a hidden markov model

    :param Observation: ndarray, shape(T,) index of observation
        T: number of observation
    :param Emission: ndarray, shape(N,M) emission proba of specific
        observation given a hidden state
            Emission[i,j] proba of observing j given the hidden state i
            N: number of hidden state
            M: number of all possible observations
    :param Transition: ndarray, shape(N,N) transition proba
        transition[i,j] proba transitioning from hidden state i to j
    :param Initial: ndarray, shape(N,1) proba of starting in a particular
        hidden state

    :return: P, F or None, None on failure
        P: likelihood of observations given model
        F: ndarray, shape(N,T) forward path proba
            F[i,j] proba being in hidden state i at time j given the
                previous observations
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None
    if (not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2
            or Initial.shape[0] != N or Initial.shape[1] != 1):
        return None, None

    alpha = np.zeros((N, T))
    # First observation using initial state
    alpha[:, 0] = Initial.reshape(-1) * Emission[:, Observation[0]]

    # For other observations
    for t in range(1, T):
        # for each hidden state
        for j in range(N):
            obs_idx = Observation[t]
            if obs_idx >= M:
                alpha[j, t] = 0  # Handle out-of-bounds observation index
            else:
                alpha[j, t] = np.dot(alpha[:, t - 1],
                                     Transition[:, j]) * Emission[j, obs_idx]

    P = np.sum(alpha[:, T - 1])

    return P, alpha
