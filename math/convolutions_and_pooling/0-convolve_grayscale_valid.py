#!/usr/bin/env python3
"""Module for valid convolution."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array with shape (m, h, w) containing multiple grayscale images.
        kernel (numpy.ndarray): Array with shape (kh, kw) containing the kernel for the convolution.

    Returns:
        numpy.ndarray: Array containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output shape
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Flip the kernel for convolution
    kernel = np.flipud(np.fliplr(kernel))

    # Iterate through each image
    for i in range(m):
        # Iterate through each pixel in the image
        for j in range(out_h):
            for k in range(out_w):
                # Extract the region of interest
                roi = images[i, j:j+kh, k:k+kw]
                # Perform element-wise multiplication and sum
                output[i, j, k] = np.sum(roi * kernel)

    return output
