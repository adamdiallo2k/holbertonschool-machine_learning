#!/usr/bin/env python3
"""Commented module """


import tensorflow as tf


def model_input(nx, classes):
    """
    Defines input and output layers for a model in TensorFlow 2.x.

    Parameters:
    nx (int): The number of feature columns in our data.
    classes (int): The number of classes in our classifier.

    Returns:
    A TensorFlow Keras model with the specified input and output layer shapes.
    """
    inputs = tf.keras.Input(shape=(nx,))
    outputs = tf.keras.layers.Dense(classes, activation='softmax')(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
