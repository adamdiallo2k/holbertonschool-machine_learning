#!/usr/bin/env python3
"""
Calculate the cost of a neural network with L2 regularization.
"""

import tensorflow as tf

def l2_reg_cost(cost):
    """
    Adds L2 regularization losses to the base cost of the neural network.

    :param cost: Tensor, the base cost (e.g., cross-entropy loss) without L2 regularization.
    :return: Tensor, the total cost including L2 regularization losses.
    """
    # Get the list of all L2 regularization losses from the model
    regularization_losses = tf.add_n(tf.keras.losses.get_regularization_losses())

    # Combine the base cost with the regularization losses
    total_cost = cost + regularization_losses

    return total_cost
