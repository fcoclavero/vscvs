__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" GAN discriminator modules for the adversarial architecture. """


import torch.nn as nn
import torch.nn.functional as F

from vscvs.models.mixins import SigmoidMixin, SoftmaxMixin


class InterModalDiscriminatorBase(nn.Module):
    """
    Fully connected network that classifies vectors in the common vector space as belonging any mode (image or sketch).
    """
    def __init__(self, input_dimension):
        """
        Initialize model.
        :param input_dimension: the size of the input vector (common vector space dimensionality).
        :type: int
        """
        super().__init__()
        self.linear_0 = nn.Linear(input_dimension, 75)
        self.linear_1 = nn.Linear(75, 25)
        self.linear_2 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_1(x))
        return x


class InterModalDiscriminatorSigmoid(SigmoidMixin, InterModalDiscriminatorBase):
    pass # 0 < output_value < 1


class InterModalDiscriminatorSoftmax(SoftmaxMixin, InterModalDiscriminatorBase):
    pass # values must sum 1 on dimension 1, that is for the 2D outputs for each example
