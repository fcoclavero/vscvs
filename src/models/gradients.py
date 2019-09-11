import torch

import torch.nn.functional as F


class AbstractKernelConvolution(torch.nn.Module):
    """ Abstract nn.Module for creating convolutions with user defined kernels. """

    @property
    def kernel(self):
        """ Get the custom kernel for the implementation of the kernel convolution. """
        raise NotImplementedError

    def forward(self, x):

        """
        Perform the convolution of the input with the Sobel kernel.
        :param x: the image batch
        :type: torch.tensor[batch_size, n_channels, x_dimension, y_dimension]
        :return: the application of the Sobel filter over the input
        :type: torch.tensor
        """
        return F.conv2d(x, self.kernel, padding=1) # zero-padding maintains image dimensions. 1 is enough given kernel

    def __call__(self, *input, **kwargs):
        """ Make the object callable by executing the `forward` function`. """
        return self.forward(*input, **kwargs)


class SobelX(AbstractKernelConvolution):
    """
    Torch nn Layer for the Sobel operator along the x axis. It approximates the gradient along the x axis by
    filtering (convolution) the input with a specific kernel. It only works with grayscale images.
    Reference: https://en.wikipedia.org/wiki/Sobel_operator
    """
    @property
    def kernel(self):
        kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        return kernel[None, None] # kernel was defined as 2D, so we must `unsqueeze(self.kernel, 0)` twice
