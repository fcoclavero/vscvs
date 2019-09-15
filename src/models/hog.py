import math
import torch

from src.models.gradients import SobelX, SobelY


class HOG(torch.nn.Module):
    def __init__(self, cell_size=8, n_bins=9, signed_gradients=False):
        """
        Model for creating a histogram of oriented gradients feature vectors for the given images.
        Does not require training.
        Reference: https://www.learnopencv.com/histogram-of-oriented-gradients/
        :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
        calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
        :type: int
        :param n_bins: number of bins for the histogram of each cell.
        :type: int
        :param signed_gradients: gradients are represented using its angle and magnitude. Angles can be expressed
        using values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the
        gradient “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has
        been shown that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
        :type: boolean
        """
        super(HOG, self).__init__()
        # Set hyperparameters
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.signed_gradients = signed_gradients
        # Define constituent layers
        self.sobel_x = SobelX()  # Sobel filtering layer
        self.sobel_y = SobelY()  # Sobel filtering layer
        self.cell_pooling = torch.nn.AvgPool2d(cell_size, stride=cell_size, padding=0)

    @property
    def angle_range(self):
        """ Range of possible gradient angles. Depends on whether signed gradients are used. """
        return 2 * math.pi if self.signed_gradients else math.pi

    def forward(self, x):
        """
        Transform the incoming image batch into HOG feature vectors.
        :param x: the image batch
        :type: torch.tensor[batch_size, n_channels, x_dimension, y_dimension]
        :return: the HOG descriptor
        :type: torch.tensor
        """
        with torch.no_grad():  # we won't need gradients for operations, so we use this option for better performance
            n_inputs, _, input_height, input_width = x.shape

            # First, we need to compute the gradients along both axes.
            gx, gy = self.sobel_x(x), self.sobel_y(x)
            grad_magnitudes, grad_angles = torch.sqrt(gx**2 + gy**2), torch.atan2(gx, gy)

            # If signed angles are used, we phase shift by pi to get only positive numbers
            grad_angles = grad_angles + math.pi if self.signed_gradients else grad_angles.abs()

            # Gradient angle linear interpolation. First we divide angles by the maximum angle. This gives us the angle
            grad_angle_interpolation = grad_angles / self.angle_range # as a fraction [0, 1] of the maximum.
            # We then multiply by 1 - n_bins and take the floor, giving us an int that corresponds to the angle
            grad_bins = (grad_angle_interpolation * (self.n_bins-  1)).floor().long() # bin the pixel belongs to.

            # Now we need the histogram for every pixel block. First, we create tensor with a vector for each pixel,
            # containing its gradient magnitude in the index of the pixel's gradient orientation bin.
            out = torch.zeros((n_inputs, self.n_bins, input_height, input_width), dtype=torch.float, device=x.device)
            out.scatter_(1, grad_bins, grad_magnitudes) # the scatter function places the mag in the corresponding index

            # Now we use an average pool with `cell_size` kernel, which gives us the normalized sum of the pixel vectors
            # above, giving us a single vector for each cell with the normalized histogram of orientations.
            hog = self.cell_pooling(out) * self.cell_size**2

            # Now we flatten to return the actual feature vector
            return hog.flatten()