from cv2 import imread, imshow, waitKey
from typing import List
from pathlib import Path
from threading import Thread


def imsshow(image_paths: List[Path]):
    """
    Displays many images.
    """
    for image_path in image_paths:
        image_path = str(image_path)
        image = imread(image_path)
        imshow(image_path, image)
    waitKey()
