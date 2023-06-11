import numpy as np
from numpy.typing import ArrayLike


def bgr_to_hsv(image: ArrayLike) -> ArrayLike:
    hsv_image = np.zeros_like(image)

    # General calculations
    c_max_idx = image.argmax(axis=2)
    c_max = image.max(axis=2)
    c_min = image.min(axis=2)
    delta = c_max - c_min

    # Masks for 4 cases of Hue
    zero_delta = delta == 0
    nonzero_delta = ~zero_delta
    c_max_r = (c_max_idx == 2) & nonzero_delta
    c_max_g = (c_max_idx == 1) & nonzero_delta
    c_max_b = (c_max_idx == 0) & nonzero_delta

    # Calculations for 4 cases of Hue
    hsv_image[zero_delta, 0] = 0
    hsv_image[c_max_r, 0] = (image[c_max_r, 1] - image[c_max_r, 0]) / delta[c_max_r] % 6
    hsv_image[c_max_g, 0] = (image[c_max_g, 0] - image[c_max_g, 2]) / delta[c_max_g] + 2
    hsv_image[c_max_b, 0] = (image[c_max_b, 2] - image[c_max_b, 1]) / delta[c_max_b] + 4
    hsv_image[:, :, 0] /= 6

    # Masks and calculations for 2 cases of Saturation
    zero_c_max = c_max == 0
    nonzero_x_max = ~zero_c_max
    hsv_image[zero_c_max, 1] = 0
    hsv_image[nonzero_x_max, 1] = delta[nonzero_x_max] / c_max[nonzero_x_max]

    # Value
    hsv_image[:, :, 2] = c_max

    return hsv_image
