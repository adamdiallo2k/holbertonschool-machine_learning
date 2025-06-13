#!/usr/bin/env python3
"""Flip an image horizontally using TensorFlow."""

import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image (tf.Tensor): 3D tensor containing the image to flip.

    Returns:
        tf.Tensor: Horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
