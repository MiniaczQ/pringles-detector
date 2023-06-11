import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def similar_sizes(
    sizes: ArrayLike, other_sizes: ArrayLike, tolerances: Tuple[float, float]
) -> ArrayLike:
    minmax = np.repeat(sizes[:, None], 2, axis=1) * np.array(tolerances)
    minmax = minmax.astype(np.int32)
    other_sizes = np.repeat(other_sizes[:, None], sizes.shape[0], axis=1)
    mask = (other_sizes >= minmax[:, 0]) & (other_sizes <= minmax[:, 1])
    return mask


def relative_positions(
    sizes: ArrayLike,
    centroids: ArrayLike,
    other_centroids: ArrayLike,
    tolerances: Tuple[Tuple[float, float], Tuple[float, float]],
) -> ArrayLike:
    centroid_count = centroids.shape[0]
    other_centroid_count = other_centroids.shape[0]
    sqrt_sizes = np.sqrt(sizes)
    minmax = np.repeat(
        np.repeat(sqrt_sizes[:, None, None], 2, axis=1), 2, axis=2
    ) * np.array(tolerances)
    minmax = minmax.astype(np.int32)
    centroids = np.repeat(centroids[:, None, :], other_centroid_count, axis=1)
    other_centroids = np.repeat(other_centroids[None, :, :], centroid_count, axis=0)
    delta = other_centroids - centroids
    mask = (delta >= minmax[:, 0, :][:, None, :]) & (
        delta <= minmax[:, 1, :][:, None, :]
    )
    return mask[:, :, 0] & mask[:, :, 1]


if __name__ == "__main__":
    sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    other_sizes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(similar_sizes(sizes, other_sizes, (0.5, 1.5)))
    centroids = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
        ]
    )
    other_centroids = np.array(
        [
            [9, 9],
            [8, 8],
            [7, 7],
            [6, 6],
            [5, 5],
            [4, 4],
            [3, 3],
            [2, 2],
            [1, 1],
            [0, 0],
        ]
    )
    print(relative_positions(sizes, centroids, other_centroids, ((-0.5, -0.5), (1, 1))))
