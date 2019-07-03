import torch

from hog.histogram import gradient, magnitude_orientation, hog, visualise_histogram


def hog_features(image, cell_size=(8, 8), cells_per_block=(1, 1), n_bins=9, signed_orientation=False, normalize=True):
    """
    Create a histogram of oriented gradients feature vector for the given image.
    Reference: https://www.learnopencv.com/histogram-of-oriented-gradients/
    :param image: the image who's feature vector is to be computed
    :type: torch.tensor
    :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
    calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
    :type: tuple<int,int>
    :param cells_per_block: used for block normalization. To make the descriptor more independent of lighting
    variations, blocks composed of the specified number of cells are used for normalization. The block normalization
    is applied in a similar fashion as a convolution.
    :type: tuple<int, int>
    :param n_bins: number of bins for the histogram of each cell.
    :type: int
    :param signed_orientation: gradients are represented using its angle and magnitude. Angles can be expressed using
    values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the gradient
    “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has been shown
    that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
    :type: boolean
    :param normalize: weather block normalization should be used or not
    :type: boolean
    :return: the HOG descriptor
    :type: torch.tensor
    """
    image = image[0][0].numpy()
    descriptor = hog(
        image,
        cell_size = cell_size,
        cells_per_block = cells_per_block,
        visualise = False,
        nbins = n_bins,
        signed_orientation = signed_orientation,
        normalise = normalize
    )
    return torch.from_numpy(descriptor).reshape(-1)