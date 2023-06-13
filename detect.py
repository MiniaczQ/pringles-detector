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
from processing.debug import dbg_repr_ccl, dbg_repr_mask
from processing.morph import erode, dilate, repopulate, populate
from processing.compare import similar_sizes, relative_positions
from processing.aabb import draw_aabbs


def detect(image: ArrayLike) -> ArrayLike:
    """
    Performs the detection pipeline.
    Returns image with marked detected icons.
    """

    # BGR to HSV
    hsv_image = bgr_to_hsv(image.astype(np.float32) / 255)

    # Extract face & text masks
    face_mask = (hsv_image[:, :, 1] < 0.15) & (hsv_image[:, :, 2] > 0.5)
    text_mask = (hsv_image[:, :, 0] < 0.2) & (hsv_image[:, :, 0] > 0.095)
    hair_mask = hsv_image[:, :, 2] < 0.4

    # Erode and dilate text mask
    text_mask = dilate(erode(text_mask, 2), 4)

    # CCL, Sizes and CoGs for face mask
    face_labels = ccl(face_mask)
    face_uniques = label_uniques(face_labels)
    face_sizes = label_sizes(face_labels, face_uniques)
    face_cogs = label_cogs(face_labels, face_uniques, face_sizes)

    # CCL, Sizes and CoGs for text mask
    text_labels = ccl(text_mask)
    text_uniques = label_uniques(text_labels)
    text_sizes = label_sizes(text_labels, text_uniques)
    text_cogs = label_cogs(text_labels, text_uniques, text_sizes)

    # Compare sizes and relative positions between text and face labels
    textface_size_mask = similar_sizes(text_sizes, face_sizes, (0.5, 1.25))
    textface_pos_mask = relative_positions(
        text_sizes,
        text_cogs,
        face_cogs,
        ((-1.7, -0.7), (-0.8, 0.4)),
    )
    textface_mask = textface_size_mask & textface_pos_mask
    textface_pairs = label_matches(textface_mask)

    # Merge matching text & face labels into a new textface label
    textface_labels = merge_labels(
        text_labels, text_uniques, face_labels, face_uniques, textface_pairs
    )
    textface_uniques = label_uniques(textface_labels)
    textface_sizes = label_sizes(textface_labels, textface_uniques)
    textface_cogs = label_cogs(textface_labels, textface_uniques, textface_sizes)

    # CCL, Sizes and CoGs for hair mask
    hair_labels = ccl(hair_mask)
    hair_uniques = label_uniques(hair_labels)
    hair_sizes = label_sizes(hair_labels, hair_uniques)
    hair_cogs = label_cogs(hair_labels, hair_uniques, hair_sizes)

    # Compare relative positions between textface and hair labels
    logo_size_mask = similar_sizes(textface_sizes, hair_sizes, (0.075, 2.0))
    logo_pos_mask = relative_positions(
        textface_sizes,
        textface_cogs,
        hair_cogs,
        ((-0.5, -0.5), (0.2, 0.5)),
    )
    logo_mask = logo_size_mask & logo_pos_mask
    logo_pairs = label_matches(logo_mask)

    # Merge matching textface & hair labels into a new logo label
    logo_labels = merge_labels(
        textface_labels, textface_uniques, hair_labels, hair_uniques, logo_pairs
    )
    logo_uniques = label_uniques(logo_labels)

    # Convert labels to AABBs and draw them on the original image
    aabbs = labels_to_aabbs(logo_labels, logo_uniques)
    image = draw_aabbs(image, aabbs, np.array([0, 255, 0]))

    return [
        image,
        dbg_repr_ccl(logo_labels) * 255
        # dbg_repr_mask(face_mask) * 255,
        # dbg_repr_mask(text_mask) * 255,
        # dbg_repr_mask(hair_mask) * 255,
    ]


if __name__ == "__main__":
    from cv2 import imread, imshow, waitKey

    image_path = "input_scaled/GettyImages-518503123.jpg"
    image = imread(image_path)
    image = detect(image) / 255
    imshow(image_path, image)
    waitKey()
