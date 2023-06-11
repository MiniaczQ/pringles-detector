import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


# Based on https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def coordinate_grid(shape: Tuple[int, int]) -> ArrayLike:
    grid = cartesian_product(np.arange(shape[0]), np.arange(shape[1]))
    return grid.reshape((shape[0], shape[1], 2))


# Connected component labeling
# Based on https://en.wikipedia.org/wiki/Connected-component_labeling
def ccl(image: ArrayLike) -> ArrayLike:
    width, height = image.shape

    # Pad to prevent out of bounds checks
    padded_image = np.pad(image, [[1, 0], [1, 0]], "constant", constant_values=0)
    grid = coordinate_grid((width, height)) + 1
    foreground = grid[image == True]

    labels = np.zeros_like(padded_image, dtype=np.uint32)
    linked = [[]]
    next_label = 1

    # First pass
    for x, y in foreground:
        has_neighbours = padded_image[x - 1][y] | padded_image[x][y - 1]
        if has_neighbours:
            neighbours = []
            if padded_image[x - 1][y]:
                neighbours.append(labels[x - 1][y])
            if padded_image[x][y - 1]:
                neighbours.append(labels[x][y - 1])
            labels[x][y] = min(neighbours)
            # Union
            if padded_image[x - 1][y] & padded_image[x][y - 1]:
                l1 = labels[x - 1][y]
                l2 = labels[x][y - 1]
                linked[l1].update(linked[l2])
                for label in linked[l1]:
                    linked[label] = linked[l1]
        else:
            linked.append(set([next_label]))
            labels[x][y] = next_label
            next_label += 1

    # Second pass
    for x, y in foreground:
        # Find
        labels[x][y] = min(linked[labels[x][y]])

    # Remove padding
    return labels[1:, 1:]


def label_uniques(labels: ArrayLike) -> ArrayLike:
    return np.unique(labels)[1:]


def label_sizes(labels: ArrayLike, label_uniques: ArrayLike) -> ArrayLike:
    sizes = np.zeros_like(label_uniques)
    for i, label in enumerate(label_uniques):
        sizes[i] = np.count_nonzero(labels == label)
    return sizes


def label_centroids(
    labels: ArrayLike, label_uniques: ArrayLike, label_sizes: ArrayLike
) -> ArrayLike:
    centroids = np.zeros((label_uniques.shape[0], 2))
    grid = coordinate_grid(labels.shape)
    for i, label in enumerate(label_uniques):
        masked = grid[labels == label]
        centroids[i] = np.sum(masked, axis=0) / label_sizes[i]
    return centroids


def label_matches(label_mask: ArrayLike) -> ArrayLike:
    grid = coordinate_grid((label_mask.shape[0], label_mask.shape[1]))
    return grid[label_mask, :]


if __name__ == "__main__":
    image = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
        ]
    )

    ccled = ccl(image)
    print(ccled)
    unique = label_uniques(ccled)
    print(unique)
    sizes = label_sizes(ccled, unique)
    print(sizes)
    centroids = label_centroids(ccled, unique, sizes)
    print(centroids)
