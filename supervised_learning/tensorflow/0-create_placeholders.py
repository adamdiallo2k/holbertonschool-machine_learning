#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    Creates and returns two placeholders, x and y, for a neural network.

    Arguments:
    nx -- scalar, size of an image vector (num_px * num_px = nx)
    classes -- scalar, number of classes (from 0 to classes-1)

    Returns:
    x -- placeholder for the data input, of shape [None, nx] and dtype "float"
    y -- placeholder for the input labels, of shape [None, classes] and dtype "float"
    """

    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y
