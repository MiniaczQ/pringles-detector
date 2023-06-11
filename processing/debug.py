# Only for debugging

import numpy as np
from numpy.typing import ArrayLike
from matplotlib.colors import hsv_to_rgb
import math


# ONLY FOR DEBUGGING
def dbg_hsv_to_bgr(hsv_image: ArrayLike) -> ArrayLike:
    return hsv_to_rgb(hsv_image)[:, :, [2, 1, 0]]


# Id-based class color using a hash function to determine hue
# hash(input) = input * K mod W
# where K = phi (golden ratio)
# and W = 2 pi
phi = (1 + math.sqrt(5)) / 2
C_CLASS = lambda id: np.array([id * phi % math.tau / math.tau, 1, 0.8])


def dbg_show_ccl(labels: ArrayLike, unique: ArrayLike | None = None) -> ArrayLike:
    if unique is None:
        unique = np.unique(labels)[1:]
    width, height = labels.shape
    out_image = np.zeros((width, height, 3))
    for i, label in enumerate(unique):
        out_image[labels == label] = C_CLASS(i)
    out_image = dbg_hsv_to_bgr(out_image)
    return out_image
