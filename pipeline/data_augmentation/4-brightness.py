#!/usr/bin/env python3
"""Randomly change the brightness of an image using TensorFlow."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to change.
        max_delta (float): Maximum delta for brightness change.

    Returns:
        tf.Tensor: Brightness-adjusted image tensor.
    """
    # tf.image.random_brightness adjusts brightness by a random factor in [-max_delta, max_delta]
    return tf.image.random_brightness(image, max_delta)
