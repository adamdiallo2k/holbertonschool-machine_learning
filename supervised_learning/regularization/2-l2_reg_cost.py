#!/usr/bin/env python3
"""
Module to calculate the cost of a neural network with L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Arguments:
        cost: tensor containing the cost of the network without L2 regularization
        model: Keras model that includes layers with L2 regularization

    Returns:
        tensor containing the regularization losses for each layer
    """
    # Get the L2 regularization losses from the model
    l2_losses = []
    
    # Collect regularization losses from each layer
    for layer in model.layers:
        # Check if the layer has regularization losses
        if hasattr(layer, 'losses') and layer.losses:
            l2_losses.extend(layer.losses)
    
    # Return the regularization losses as a tensor
    if l2_losses:
        return tf.convert_to_tensor(l2_losses)
    else:
        return tf.constant(0.0)
