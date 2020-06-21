__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" GAN discriminator modules for the adversarial architecture. """


import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides

from vscvs.models.mixins import OutFeaturesMixin
from vscvs.models.mixins import SigmoidMixin
from vscvs.models.mixins import SoftmaxMixin


class InterModalDiscriminatorBase(nn.Module):
    """
    Fully connected network that classifies vectors in the common vector space as belonging any mode (image or sketch).
    """

    def __init__(self, input_dimension=None):
        """
        :param input_dimension: the size of the input vector (common vector space dimensionality).
        :type: int
        """
        super().__init__()
        self.linear_0 = nn.Linear(input_dimension, 75)
        self.linear_1 = nn.Linear(75, 25)

    @overrides
    def forward(self, x):
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        return x


class InterModalDiscriminator(OutFeaturesMixin, InterModalDiscriminatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, out_features=1, **kwargs)

    @overrides
    def forward(self, x):
        return super().forward(x).view(-1)


class InterModalDiscriminatorSigmoid(SigmoidMixin, InterModalDiscriminatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, out_features=1, **kwargs)

    @overrides
    def forward(self, x):
        return super().forward(x).view(-1)


class InterModalDiscriminatorSoftmax(SoftmaxMixin, OutFeaturesMixin, InterModalDiscriminatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, out_features=2, **kwargs)
