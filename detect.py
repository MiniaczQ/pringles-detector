import numpy as np
from numpy.typing import ArrayLike
from processing.convert import bgr_to_hsv
from processing.ccl import ccl
from processing.debug import dbg_show_ccl


def process_face(hsv_image: ArrayLike) -> ArrayLike:
    face_mask = (hsv_image[:, :, 1] < 0.15) & (hsv_image[:, :, 2] > 0.5)
    return face_mask


def process_hair(hsv_image: ArrayLike) -> ArrayLike:
    hair_mask = (
        (hsv_image[:, :, 0] < 0.11)
        & (hsv_image[:, :, 0] > 0.05)
        & (hsv_image[:, :, 1] > 0.1)
        & (hsv_image[:, :, 2] > 0.2)
        & (hsv_image[:, :, 2] < 0.95)
    )
    return hair_mask


def process_text(hsv_image: ArrayLike) -> ArrayLike:
    text_mask = (hsv_image[:, :, 0] < 0.2) & (hsv_image[:, :, 0] > 0.1)
    return text_mask


def detect(image: ArrayLike) -> ArrayLike:
    image = image.astype(np.float32)
    image /= 255
    hsv_image = bgr_to_hsv(image)

    face_mask = process_face(hsv_image)
    hair_mask = process_hair(hsv_image)
    text_mask = process_text(hsv_image)

    return dbg_show_ccl(ccl(hair_mask))

    mask = face_mask  # face_mask | hair_mask | text_mask

    image[~mask] = 0

    return image


if __name__ == "__main__":
    from cv2 import imread, imshow, waitKey

    image_path = "input_scaled/GettyImages-518503123.jpg"
    image = imread(image_path)
    image = detect(image)
    imshow(image_path, image)
    waitKey()
