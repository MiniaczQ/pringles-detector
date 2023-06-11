import numpy as np
from numpy.typing import ArrayLike


# Based on https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


# Connected component labeling
# Based on https://en.wikipedia.org/wiki/Connected-component_labeling
def ccl(image: ArrayLike) -> ArrayLike:
    width, height = image.shape

    # Pad to prevent out of bounds checks
    padded_image = np.pad(image, [[1, 0], [1, 0]], "constant", constant_values=0)
    grid = cartesian_product(np.arange(width), np.arange(height)) + 1
    grid = grid.reshape((width, height, 2))
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


if __name__ == "__main__":
    image = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
        ]
    )

    print(ccl(image))
