__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


import math
import torch

from overrides import overrides

from vscvs.models.gradients import SobelX, SobelY
from vscvs.decorators import torch_no_grad


class HOG(torch.nn.Module):
    def __init__(self, in_channels=3, cell_size=8, bins=9, signed_gradients=False):
        """
        Model for creating histogram of oriented gradients feature vectors for the given images. Doesn't need training.
        Reference: https://www.learnopencv.com/histogram-of-oriented-gradients/
        :param in_channels: the number of channels for inputs.
        :type: int
        :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
        calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
        :type: int
        :param bins: number of bins for the histogram of each cell.
        :type: int
        :param signed_gradients: gradients are represented using its angle and magnitude. Angles can be expressed
        using values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the
        gradient “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has
        been shown that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
        :type: bool
        """
        super().__init__()
        # Set hyperparameters
        self.cell_size = cell_size
        self.bins = bins
        self.signed_gradients = signed_gradients
        # Define constituent layers
        self.sobel_x = SobelX(in_channels=in_channels)  # Sobel filtering layer
        self.sobel_y = SobelY(in_channels=in_channels)  # Sobel filtering layer
        self.cell_pooling = torch.nn.AvgPool2d(cell_size)

    @property
    def angle_range(self):
        """
        Range of possible gradient angles. Depends on whether signed gradients are used.
        """
        return 2 * math.pi if self.signed_gradients else math.pi

    def descriptor_length(self, in_dimension):
        """
        Get the length of output descriptors, given the dimensions of the would be inputs.
        :param in_dimension: input image dimensions (assuming square images).
        :type: int
        """
        return int(self.bins * (in_dimension / self.cell_size) ** 2)

    @torch_no_grad # we won't need gradients for operations, so we use this option for better performance
    @overrides
    def forward(self, x):
        n_inputs, _, input_height, input_width = x.shape
        # First, we need to compute the gradients along both axes.
        gx, gy = self.sobel_x(x), self.sobel_y(x)
        grad_magnitudes, grad_angles = torch.sqrt(gx**2 + gy**2), torch.atan2(gx, gy)
        # If signed angles are used, we phase shift by pi to get only positive numbers
        grad_angles = grad_angles + math.pi if self.signed_gradients else grad_angles.abs()
        # Gradient angle linear interpolation. First we divide angles by the maximum angle. This gives us the angle
        grad_angle_interpolation = grad_angles / self.angle_range # as a fraction [0, 1] of the maximum.
        # We then multiply by 1 - bins and take the floor, giving us an int that corresponds to the angle
        grad_bins = (grad_angle_interpolation * (self.bins-  1)).floor().long() # bin the pixel belongs to.
        # Now we need the histogram for every pixel block. First, we create tensor with a vector for each pixel,
        # containing its gradient magnitude in the index of the pixel's gradient orientation bin.
        out = torch.zeros((n_inputs, self.bins, input_height, input_width), dtype=torch.float, device=x.device)
        out.scatter_(1, grad_bins, grad_magnitudes) # the scatter function places the mag in the corresponding index
        # Now we use an average pool with `cell_size` kernel, which gives us the normalized sum of the pixel vectors
        # above, giving us a single vector for each cell with the normalized histogram of orientations.
        hog = self.cell_pooling(out) * self.cell_size**2
        # Now we flatten to return the actual feature vector
        return hog.flatten(start_dim=1) # start_dim=1 to return the hog vector of every image in a batch

    @overrides
    def to(self, *args, **kwargs):
        self.sobel_x, self.sobel_y = self.sobel_x.to(*args, **kwargs), self.sobel_y.to(*args, **kwargs)
        return super().to(*args, **kwargs)
