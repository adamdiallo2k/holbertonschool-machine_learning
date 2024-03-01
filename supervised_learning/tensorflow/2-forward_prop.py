#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Import the create_layer function
create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Arguments:
    x -- the placeholder for the input data.
    layer_sizes -- list containing the number of nodes in each layer of the network.
    activations -- list containing the activation functions for each layer of the network.

    Returns:
    The prediction of the network in tensor form.
    """

    # Initialize the input for the first layer
    a = x

    # Iterate through each layer to create the network
    for i, size in enumerate(layer_sizes):
        
        activation = activations[i] if i < len(activations) else None

        # Create the layer and update 'a' to be the output of the current layer
        a = create_layer(a, size, activation)

    # 'a' now holds the output of the last layer, which is the network's prediction
    return a
