#!/usr/bin/env python3
"""
Module 4-calculate_loss

Provides a function `calculate_loss` that calculates the softmax
cross-entropy loss of a prediction.
"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Args:
        y (tensor): Placeholder for the labels of the input data.
        y_pred (tensor): Tensor containing the network’s predictions (logits).

    Returns:
        tensor: A tensor containing the loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
