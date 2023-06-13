import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def within(aabb1: ArrayLike, aabb2: ArrayLike) -> bool:
    """
    Whether AABB1 is contained within AABB2.
    """
    return (
        (aabb2[0] <= aabb1[0])
        & (aabb2[1] <= aabb1[1])
        & (aabb1[2] <= aabb2[2])
        & (aabb1[3] <= aabb2[3])
    )


def remove_overlaps(aabbs: ArrayLike) -> ArrayLike:
    """
    Removes smaller AABBs when there are multiple within one another.
    """
    valid = set(range(aabbs.shape[0]))
    for aabb_big in aabbs:
        for i, aabb_small in enumerate(aabbs):
            if not (aabb_big == aabb_small).all():
                if i in valid:
                    if within(aabb_small, aabb_big):
                        valid.remove(i)
    return aabbs[list(valid), :]


def draw_aabbs(image: ArrayLike, aabbs: ArrayLike, color: ArrayLike) -> ArrayLike:
    """
    Draws many AABBs onto an image.
    """
    for x_min, y_min, x_max, y_max in aabbs:
        image[[x_min, x_max], y_min : y_max + 1] = color
        image[x_min : x_max + 1, [y_min, y_max]] = color
    return image


if __name__ == "__main__":
    from cv2 import imshow, waitKey
    import numpy as np

    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    image = draw_aabbs(image, np.array([[15, 15, 31, 31]]), np.array([0, 0, 0]))
    imshow("image", image)
    waitKey()
