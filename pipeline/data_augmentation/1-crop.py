#!/usr/bin/env python3
"""Randomly crop an image using TensorFlow."""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image (tf.Tensor): 3D tensor containing the image to crop.
        size (tuple): Tuple of integers (crop_height, crop_width, channels).

    Returns:
        tf.Tensor: Randomly cropped image tensor.
    """
    return tf.image.random_crop(image, size)
