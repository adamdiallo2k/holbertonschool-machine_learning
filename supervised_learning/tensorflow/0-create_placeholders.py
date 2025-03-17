#!/usr/bin/env python3
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network:
      - x is placeholder for the input data to the network
      - y is placeholder for the one-hot labels
    :param nx: number of feature columns in the data
    :param classes: number of classes in the classifier
    :return: placeholders x and y
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
