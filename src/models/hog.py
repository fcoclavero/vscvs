import torch

from hog.histogram import hog


class HOG:
    def __init__(self, cell_size=(8, 8), cells_per_block=(1, 1), n_bins=9, signed_orientation=False, normalize=True):
        """
        Model for creating a histogram of oriented gradients feature vectors for the given images.
        Does not require training.
        Reference: https://www.learnopencv.com/histogram-of-oriented-gradients/
        :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
        calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
        :type: tuple<int,int>
        :param cells_per_block: used for block normalization. To make the descriptor more independent of lighting
        variations, blocks composed of the specified number of cells are used for normalization. The block normalization
        is applied in a similar fashion as a convolution.
        :type: tuple<int, int>
        :param n_bins: number of bins for the histogram of each cell.
        :type: int
        :param signed_orientation: gradients are represented using its angle and magnitude. Angles can be expressed
        using values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the
        gradient “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has
        been shown that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
        :type: boolean
        :param normalize: weather block normalization should be used or not
        :type: boolean
        """
        self.cell_size = cell_size
        self.cells_per_block = cells_per_block
        self.n_bins = n_bins
        self.signed_orientation = signed_orientation
        self.normalize = normalize

    def forward(self, x):
        """
        Transform the incoming image batch into HOG feature vectors.
        :param x: the image batch
        :type: torch.tensor[batch_size, n_channels, x_dimension, y_dimension]
        :return: the HOG descriptor
        :type: torch.tensor
        """
        x = x[0][0].numpy()
        descriptor = hog(
            x,
            cell_size = self.cell_size,
            cells_per_block = self.cells_per_block,
            visualise = False,
            nbins = self.n_bins,
            signed_orientation = self.signed_orientation,
            normalise = self.normalize
        )
        return torch.from_numpy(descriptor).reshape(-1)

    def __call__(self, *input, **kwargs):
        """ Make the object callable by executing the `forward` function`. """
        return self.forward(*input, **kwargs)