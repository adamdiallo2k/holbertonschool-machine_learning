#!/usr/bin/env python3
"""
Module with a function to create a dropout layer in a neural network
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout

    Arguments:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        output of the new layer
    """
    # Initialize weights using He et al. method
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')

    # Create the layer with the specified number of nodes
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )

    # Apply the layer to the previous tensor
    output = layer(prev)

    # Apply dropout regularization
    dropout_output = tf.keras.layers.Dropout(
        rate=1 - keep_prob
    )(output, training=training)

    return dropout_output
