#!/usr/bin/env python3
"""
Module 1-create_layer

Provides a function `create_layer` that builds a layer for the neural network.
"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates a TensorFlow layer using tf.keras.initializers.VarianceScaling with
    mode 'fan_avg' for the kernel weights.

    Args:
        prev (tensor): Tensor output of the previous layer.
        n (int): Number of nodes in the layer to create.
        activation (function): Activation function to be used in the layer.

    Returns:
        tensor: The tensor output of the created layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
