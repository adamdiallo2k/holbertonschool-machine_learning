#!/usr/bin/env python3
"""Rotate an image by 90 degrees counter-clockwise using TensorFlow."""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): 3D tensor containing the image to rotate.

    Returns:
        tf.Tensor: Rotated image tensor.
    """
    # tf.image.rot90 rotates counter-clockwise by 90 degrees per k
    return tf.image.rot90(image, k=1)
