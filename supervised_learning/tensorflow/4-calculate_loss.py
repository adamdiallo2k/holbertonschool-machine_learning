#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Arguments:
    y -- a placeholder for the labels of the input data.
    y_pred -- a tensor containing the network’s predictions (logits).

    Returns:
    A tensor containing the loss of the prediction.
    """

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
    (logits=y_pred, labels=y))

    return loss
