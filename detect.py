import numpy as np
from numpy.typing import ArrayLike
from processing.convert import bgr_to_hsv
from processing.ccl import (
    ccl,
    label_uniques,
    label_sizes,
    label_centroids,
    label_matches,
)
from processing.debug import dbg_show_ccl
from processing.morph import erode, dilate, repopulate, populate
from processing.compare import similar_sizes, relative_positions


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
    )  # | ((hsv_image[:, :, 2] < 0.2))
    return hair_mask


def process_text(hsv_image: ArrayLike) -> ArrayLike:
    text_mask = (hsv_image[:, :, 0] < 0.2) & (hsv_image[:, :, 0] > 0.1)
    return text_mask


def detect(image: ArrayLike) -> ArrayLike:
    hsv_image = bgr_to_hsv(image.astype(np.float32) / 255)

    face_mask = process_face(hsv_image)
    hair_mask = process_hair(hsv_image)
    text_mask = process_text(hsv_image)

    face_mask_labels = ccl(face_mask)
    face_mask_uniques = label_uniques(face_mask_labels)
    face_mask_sizes = label_sizes(face_mask_labels, face_mask_uniques)
    face_mask_centroids = label_centroids(
        face_mask_labels, face_mask_uniques, face_mask_sizes
    )
    # return dbg_show_ccl(face_mask_labels) * 255

    text_mask = dilate(erode(text_mask, 2), 4)
    text_mask_labels = ccl(text_mask)
    text_mask_uniques = label_uniques(text_mask_labels)
    text_mask_sizes = label_sizes(text_mask_labels, text_mask_uniques)
    text_mask_centroids = label_centroids(
        text_mask_labels, text_mask_uniques, text_mask_sizes
    )
    # return dbg_show_ccl(text_mask_labels, text_mask_uniques[text_mask_sizes > 200]) * 255

    size_label_mask = similar_sizes(face_mask_sizes, text_mask_sizes, (0.8, 2))
    position_label_mask = relative_positions(
        text_mask_sizes, text_mask_centroids, face_mask_centroids, ((-1, -4), (1, 0))
    )
    label_mask = size_label_mask & position_label_mask
    matches = label_matches(label_mask)
    matches_mask = np.zeros_like(text_mask)
    for i, (text_label, face_label) in enumerate(matches):
        print(i, text_label, face_label)
        matches_mask[text_mask_labels == text_label] = i + 1
        matches_mask[face_mask_labels == face_label] = i + 1

    return dbg_show_ccl(matches_mask) * 255

    mask = text_mask  # face_mask | hair_mask | text_mask

    image[~mask] = 0

    return image


if __name__ == "__main__":
    from cv2 import imread, imshow, waitKey

    image_path = "input_scaled/GettyImages-518503123.jpg"
    image = imread(image_path)
    image = detect(image) / 255
    imshow(image_path, image)
    waitKey()
