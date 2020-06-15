__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Modules with image processing convolutions with fixed kernels, such as the Sobel filter. """


from abc import abstractmethod

import torch
import torch.nn.functional as F

from overrides import overrides

from vscvs.decorators import torch_no_grad


class AbstractKernelConvolution(torch.nn.Module):
    """
    Abstract module for creating convolutions with user defined kernels.
    """

    def __init__(self, in_channels=3, stride=1, padding=1, dilation=1):
        """
        :param in_channels: the number of channels for inputs.
        :type: int
        :param stride: controls the stride for the cross-correlation.
        :type: int or Tuple[int, int] with height and width dimensions, respectively
        :param padding: controls the amount of implicit zero-paddings on both sides for `padding` number of points for
        each dimension. Defaults to one to maintain input dimensions.
        :type: int or Tuple[int, int] with height and width dimensions, respectively
        :param dilation: controls the spacing between the kernel points; also known as the à trous algorithm. It is
        harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a
        nice visualization of what dilation does.
        :type: int or Tuple[int, int] with height and width dimensions, respectively
        """
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.register_buffer(
            "weight", self.kernel
        )  # register buffer that should not to be considered a model parameter

    @property
    def kernel(self):
        """ Get the custom kernel for the implementation of the kernel convolution. """
        raise NotImplementedError

    @torch_no_grad  # we won't need the gradient, so we use this option for better performance
    @overrides
    def forward(self, x):
        """
        Perform the convolution of the input with the Sobel kernel.
        :param x: the image batch
        :type: torch.Tensor[batch_size, in_channels, x_dimension, y_dimension]
        :return: the application of the Sobel filter over the input
        :type: torch.Tensor
        """
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def to(self, *args, **kwargs):
        """ Override of `to` method to send the weight buffer to the new device. """
        self.register_buffer("weight", self.kernel.to(*args, **kwargs))
        return super().to(*args, **kwargs)


class Abstract2DKernelConvolution(AbstractKernelConvolution):
    """
    Abstract module for performing 2D kernel convolutions over all input channels.
    """

    @property
    @abstractmethod
    def kernel_2d(self):
        """
        The 2D kernel for the convolution of the corresponding axis.
        :return: the 2D kernel for corresponding axis.
        :type: torch.Tensor
        """
        pass

    @property
    def kernel(self):
        kernel = torch.stack([self.kernel_2d for _ in range(self.in_channels)])  # repeat kernel for each input channel
        return kernel.unsqueeze(0)  # we must add and additional dimension to handle multiple inputs


class SobelX(Abstract2DKernelConvolution):
    """
    Torch nn Layer for the Sobel operator along the x axis. It approximates the gradient along the x axis by
    filtering (convolution) the input with a specific kernel. The gradients are computed for all input channels using
    the same 2D kernel over all of them. Sobel kernel size 1.
    Reference: https://en.wikipedia.org/wiki/Sobel_operator
    """

    @property
    @overrides
    def kernel_2d(self):
        return torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)


class SobelY(Abstract2DKernelConvolution):
    """
    Torch nn Layer for the Sobel operator along the y axis. It approximates the gradient along the y axis by
    filtering (convolution) the input with a specific kernel. The gradients are computed for all input channels using
    the same 2D kernel over all of them. Sobel kernel size 1.
    Reference: https://en.wikipedia.org/wiki/Sobel_operator
    """

    @property
    @overrides
    def kernel_2d(self):
        return torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)


class Laplacian(Abstract2DKernelConvolution):
    """
    Torch nn Layer for the Laplacian derivative operator. It approximates the gradient magnitude along both axes by
    filtering (convolution) the input with a specific kernel. The gradients are computed for all input channels using
    the same 2D kernel over all of them. Kernel size 1.
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    """

    @property
    @overrides
    def kernel_2d(self):
        return torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float)
