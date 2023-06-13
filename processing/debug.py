# Only for debugging

import numpy as np
from numpy.typing import ArrayLike
from matplotlib.colors import hsv_to_rgb
import math


def dbg_hsv_to_bgr(hsv_image: ArrayLike) -> ArrayLike:
    """
    Converts a HSV color space image to BGR color space.
    """
    return hsv_to_rgb(hsv_image)[:, :, [2, 1, 0]]


phi = (1 + math.sqrt(5)) / 2


def unique_color(id: int) -> ArrayLike:
    """
    Color from sequential id assignment using a modulo hash function to determine hue.
    """
    # hash(input) = input * K mod W
    # where K = phi (golden ratio)
    # and W = 2 pi
    return np.array([id * phi % math.tau / math.tau, 1, 0.8])


def dbg_repr_ccl(labels: ArrayLike, unique: ArrayLike | None = None) -> ArrayLike:
    """
    Converts a labeled mask to an image representation with unique color for each label.
    """
    if unique is None:
        unique = np.unique(labels)[1:]
    width, height = labels.shape
    out_image = np.zeros((width, height, 3))
    for i, label in enumerate(unique):
        out_image[labels == label] = unique_color(i)
    out_image = dbg_hsv_to_bgr(out_image)
    return out_image


def dbg_repr_mask(mask: ArrayLike) -> ArrayLike:
    """
    Converts a boolean mask to black and white image representation.
    """
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    image[mask] = 1
    return image
