__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Simple convolutional neural networks implemented from scratch. """


import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides

from vscvs.models.mixins import LogSoftmaxMixin
from vscvs.models.mixins import NormalizedMixin
from vscvs.models.mixins import OutFeaturesMixin
from vscvs.models.mixins import SigmoidMixin
from vscvs.models.mixins import SoftmaxMixin


class CNNBase(nn.Module):
    """
    Base model for a simple convolutional neural network.
    """

    def __init__(self):
        super().__init__()  # 256x256x3
        self.convolution_0 = nn.Conv2d(3, 6, 5)  # 252x252x6
        self.convolution_1 = nn.Conv2d(6, 16, 5)  # 122x122x16
        self.convolution_2 = nn.Conv2d(16, 20, 4)  # 58x58x20
        self.fully_connected_0 = nn.Linear(20 * 29 * 29, 15000)  # 15000
        self.fully_connected_1 = nn.Linear(15000, 1000)  # 1000

    @overrides
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.convolution_0(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.convolution_1(x)), 2)
        x = F.max_pool2d(F.relu(self.convolution_2(x)), 2)
        x = x.view(-1, self.number_of_flat_features(x))
        x = F.relu(self.fully_connected_0(x))
        x = self.fully_connected_1(x)
        return x

    @staticmethod
    def number_of_flat_features(x):
        """
        Get the number of features all dimensions except the batch dimension.
        :param x: the tensor.
        :type: torch.Tensor
        :return: the number of features.
        :type: int
        """
        size = x.size()[1:]  # exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN(OutFeaturesMixin, CNNBase):
    pass


class CNNNormalized(NormalizedMixin, CNNBase):
    pass


class CNNSigmoid(SigmoidMixin, CNNBase):
    pass


class CNNSoftmax(SoftmaxMixin, CNNBase):
    pass


class CNNLogSoftmax(LogSoftmaxMixin, CNNBase):
    pass
