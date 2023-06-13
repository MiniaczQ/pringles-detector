import numpy as np
from numpy.typing import ArrayLike


def erode(mask: ArrayLike, k=1) -> ArrayLike:
    """
    Starting from a blank mask, set entry to True if it has all neighbours (in the provided mask) in 2K+1 surrounding square.
    """
    padded_mask = np.pad(mask, [[k, k], [k, k]], "constant", constant_values=0)
    width, height = padded_mask.shape
    eroded_mask = np.zeros_like(padded_mask)
    for x in range(k, width - k):
        for y in range(k, height - k):
            if (padded_mask[x - k : x + k, y - k : y + k] == True).all():
                eroded_mask[x, y] = 1
    return eroded_mask[k:-k, k:-k]


def dilate(mask: ArrayLike, k=1) -> ArrayLike:
    """
    Starting from a blank mask, set entry to True if it has at least 1 neighbour (in the provided mask) in 2K+1 surrounding square.
    """
    padded_mask = np.pad(mask, [[k, k], [k, k]], "constant", constant_values=0)
    width, height = padded_mask.shape
    dilated_mask = np.zeros_like(padded_mask)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if (padded_mask[x - k : x + k, y - k : y + k] == True).any():
                dilated_mask[x, y] = 1
    return dilated_mask[k:-k, k:-k]


def repopulate(mask: ArrayLike, n=3, k=1) -> ArrayLike:
    """
    Starting from a blank mask, set entry to True if it has at least N neighbours (in the provided mask) in 2K+1 surrounding square.
    """
    padded_mask = np.pad(mask, [[k, k], [k, k]], "constant", constant_values=0)
    width, height = padded_mask.shape
    repopulated_mask = np.zeros_like(padded_mask)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if np.count_nonzero(padded_mask[x - k : x + k, y - k : y + k] == True) >= n:
                repopulated_mask[x, y] = 1
    return repopulated_mask[k:-k, k:-k]


def populate(mask: ArrayLike, n=3, k=1) -> ArrayLike:
    """
    Starting from the provided mask, set entry to True if it has at least N neighbours in 2K+1 surrounding square.
    """
    padded_mask = np.pad(mask, [[k, k], [k, k]], "constant", constant_values=0)
    width, height = padded_mask.shape
    populated_mask = padded_mask.copy()
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if np.count_nonzero(padded_mask[x - k : x + k, y - k : y + k] == True) >= n:
                populated_mask[x, y] = 1
    return populated_mask[k:-k, k:-k]
