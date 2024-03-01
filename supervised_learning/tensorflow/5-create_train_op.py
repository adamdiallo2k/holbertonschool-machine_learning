#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Arguments:
    - loss: The loss of the network’s prediction, a TensorFlow tensor.
    - alpha: The learning rate, a float.

    Returns:
    - A TensorFlow operation that trains the network using gradient descent.
    """
    # Initialize the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Create the training operation
    train_op = optimizer.minimize(loss)

    return train_op
