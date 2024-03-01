#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Creates a new TensorFlow layer.

    Arguments:
    prev -- tensor containing the output of the previous layer
    n -- integer, number of nodes in the layer to create
    activation -- the activation function to be used on the layer

    Returns:
    The tensor output of the layer.
    """

    # Define the He et al. initializer for the layer weights
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the layer with the specified number of nodes and activation function
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer(prev)
