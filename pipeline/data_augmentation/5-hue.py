#!/usr/bin/env python3
"""Change the hue of an image using TensorFlow."""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to change.
        delta (float): Amount to change the hue. Must be in [-0.5, 0.5].

    Returns:
        tf.Tensor: Hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)
