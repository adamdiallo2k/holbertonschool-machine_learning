#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Import the create_layer function
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.
    """

    a = x  # Correctly spaced comment and code

    # Adjusting the loop to avoid long lines
    for i, size in enumerate(layer_sizes):
        activation = None
        if i < len(activations):
            activation = activations[i]

        # Splitting the function call across multiple lines to address line length
        a = create_layer(
            prev=a, 
            n=size, 
            activation=activation
        )

    return a
