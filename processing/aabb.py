import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def draw_aabbs(image: ArrayLike, aabbs: ArrayLike, color: ArrayLike) -> ArrayLike:
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
