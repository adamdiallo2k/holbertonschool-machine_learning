#!/usr/bin/env python3
"""
Creates a dense layer in TensorFlow 2.x that includes L2 regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense layer with L2 regularization

    Args:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes in the new layer
        activation: the activation function to be used
        lambtha: the L2 regularization parameter (float)

    Returns:
        The output tensor of the new layer
    """
    # Define the L2 regularization
    l2_regularizer = tf.keras.regularizers.l2(lambtha)

    # Create the Dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_regularizer,
        # Optionally, specify a kernel initializer (e.g., VarianceScaling)
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    )

    return layer(prev)
