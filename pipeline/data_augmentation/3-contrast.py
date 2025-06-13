#!/usr/bin/env python3
"""Randomly adjust the contrast of an image using TensorFlow."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to adjust.
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: Contrast-adjusted image tensor.
    """
    # tf.image.random_contrast applies a random contrast change
    return tf.image.random_contrast(image, lower=lower, upper=upper)
