#!/usr/bin/env python3
"""Commented module """


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y -- a placeholder for the labels of the input data.
    y_pred -- a tensor containing the network’s predictions.

    Returns:
    A tensor containing the decimal accuracy of the prediction.
    """
    # Convert y_pred to a tensor of the same type as y
    y_pred_cls = tf.argmax(y_pred, 1)
    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
