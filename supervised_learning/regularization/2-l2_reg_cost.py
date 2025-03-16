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
        tensor containing the total cost including L2 regularization
    """
    # Get the L2 regularization losses from the model
    l2_losses = []
    for layer in model.layers:
        # Check if the layer has regularization losses
        if hasattr(layer, 'losses') and layer.losses:
            l2_losses.extend(layer.losses)
    
    # If there are L2 regularization losses, add them to the original cost
    if l2_losses:
        # Sum all the regularization losses
        l2_loss = tf.add_n(l2_losses)
        # Add the regularization loss to the original cost
        total_cost = cost + l2_loss
    else:
        total_cost = cost
    
    return total_cost
