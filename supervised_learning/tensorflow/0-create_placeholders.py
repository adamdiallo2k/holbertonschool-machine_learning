#!/usr/bin/env python3
"""
Module 0-create_placeholders

Provides a function `create_placeholders` that returns two TensorFlow
placeholders.
"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network:
      - x is placeholder for the input data to the network
      - y is placeholder for the one-hot labels

    Args:
        nx (int): number of feature columns in our data
        classes (int): number of classes in our classifier

    Returns:
        tuple: placeholders named x and y, respectively
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
