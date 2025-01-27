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
    # Sum all regularization losses
    regularization_losses = tf.add_n(tf.keras.losses.get_regularization_losses())

    # Combine the base cost with regularization losses
    total_cost = cost + regularization_losses

    return total_cost


# Main function to test the L2 regularization cost
if __name__ == "__main__":
    # Define a simple model with L2 regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

    # Example input: 3 samples, 64 features each
    x = tf.random.normal((3, 64))

    # Example labels: 3 class labels
    y = tf.random.uniform((3,), maxval=10, dtype=tf.int32)

    # Forward pass through the model
    predictions = model(x, training=True)

    # Compute base cost (categorical cross-entropy loss)
    base_cost = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, predictions))

    # Compute the total cost with L2 regularization
    total_cost = l2_reg_cost(base_cost)

    # Print the total cost
    print(total_cost)
