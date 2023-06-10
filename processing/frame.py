from numpy.typing import ArrayLike
from typing import Tuple


def frame(image: ArrayLike, rect: Tuple[Tuple[int, int], Tuple[int, int]]) -> ArrayLike:
    (x1, y1), (x2, y2) = rect
    image[[x1, x2], y1 : y2 + 1, :] = 0
    image[x1 : x2 + 1, [y1, y2], :] = 0
    return image


if __name__ == "__main__":
    from cv2 import imshow, waitKey
    import numpy as np

    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    image = frame(image, ((15, 15), (31, 31)))
    imshow("image", image)
    waitKey()
