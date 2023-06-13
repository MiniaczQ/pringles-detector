import numpy as np
from numpy.typing import ArrayLike
from processing.convert import bgr_to_hsv
from processing.labels import (
    ccl,
    label_uniques,
    label_sizes,
    label_cogs,
    label_matches,
    merge_labels,
    labels_to_aabbs,
)
from processing.debug import dbg_show_ccl, dbg_show_mask
from processing.morph import erode, dilate, repopulate, populate
from processing.compare import similar_sizes, relative_positions
from processing.aabb import draw_aabbs


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
    # hair_mask = process_hair(hsv_image)
    text_mask = process_text(hsv_image)

    face_labels = ccl(face_mask)
    face_uniques = label_uniques(face_labels)
    face_sizes = label_sizes(face_labels, face_uniques)
    face_cogs = label_cogs(face_labels, face_uniques, face_sizes)

    text_mask = dilate(erode(text_mask, 2), 4)
    text_labels = ccl(text_mask)
    text_uniques = label_uniques(text_labels)
    text_sizes = label_sizes(text_labels, text_uniques)
    text_cogs = label_cogs(text_labels, text_uniques, text_sizes)

    textface_size_mask = similar_sizes(face_sizes, text_sizes, (0.8, 2))
    textface_pos_mask = relative_positions(
        text_sizes,
        text_cogs,
        face_cogs,
        ((-1.7, -0.4), (-0.8, 0.4)),
    )
    textface_mask = textface_size_mask & textface_pos_mask
    textface_pairs = label_matches(textface_mask)

    textface_labels = merge_labels(
        text_labels, text_uniques, face_labels, face_uniques, textface_pairs
    )
    textface_uniques = label_uniques(textface_labels)

    # return dbg_show_ccl(textface_mask) * 255

    aabbs = labels_to_aabbs(textface_labels, textface_uniques)

    image = draw_aabbs(image, aabbs, np.array([0, 0, 0]))

    return image


if __name__ == "__main__":
    from cv2 import imread, imshow, waitKey

    image_path = "input_scaled/GettyImages-518503123.jpg"
    image = imread(image_path)
    image = detect(image) / 255
    imshow(image_path, image)
    waitKey()
