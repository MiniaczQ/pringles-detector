import numpy as np
from numpy.typing import ArrayLike


def detect(image: ArrayLike) -> ArrayLike:
    # TODO

    return image


if __name__ == "__main__":
    from cv2 import imread, imshow, waitKey

    image_path = "input/GettyImages-518503123.jpg"
    image = imread(image_path)
    detect(image)
    imshow(image_path, image)
    waitKey()
