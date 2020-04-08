__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixin for adapting a convolutional model for classification. """


import torch.nn.functional as F

from torch import nn, sigmoid, Tensor
from typing import Callable

from vscvs.utils import get_out_features_from_model


class ModuleMixin:
    """
    Utility class that type hints `Module` methods that will be available to the mixins in this package via a `super()`
    call, as they are meant to be used in multiple inheritance with `torch.nn.Module`.
    """
    forward: Callable[[Tensor], Tensor]


class OutFeaturesMixin(ModuleMixin):
    """
    Modifies the base model with an additional fully connected layer to match the desired output features.
    The mixin must be inherited before the `torch.nn.Module` to be extended in order to remove the additional
    `out_features` parameter.
    """
    def __init__(self, *args, out_features=None, **kwargs):
        """
        Initialize model.
        :param out_features: number of output features.
        :type: int
        """
        super().__init__(*args, **kwargs)
        self.fully_connected = nn.Linear(get_out_features_from_model(self), out_features)

    def forward(self, x):
        x = F.relu(super().forward(x))
        x = self.fully_connected(x)
        return x


class NormalizedMixin(OutFeaturesMixin):
    """
    Modifies the base model by adding a fully connected layer the same size as the number of possible classes, as well
    as normalizing the output of extended model.
    The mixin must be inherited before the `torch.nn.Module` to be extended in order to remove the additional
    `out_features` parameter used in the `OutFeaturesMixin` constructor.
    """
    def __init__(self, *args, p=2, dim=1, eps=1e-12, **kwargs):
        """
        Initialize model.
        :param p: the exponent value in the norm formulation. Default: 2
        :type: float
        :param dim: the dimension to reduce. Default: 1
        :type: int
        :param eps: small value to avoid division by zero. Default: 1e-12
        :type: float
        """
        super().__init__(*args, **kwargs)
        self.p, self.dim, self.eps = p, dim, eps

    def forward(self, x):
        x = super().forward(x)
        x = F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
        return x


class SigmoidMixin(OutFeaturesMixin):
    """
    Modifies the base model by adding a fully connected layer the same size as the number of possible classes, as well
    as a log-softmax activation function after the output of extended model.
    The mixin must be inherited before the `torch.nn.Module` to be extended in order to remove the additional
    `out_features` parameter used in the `OutFeaturesMixin` constructor.
    """
    def forward(self, x):
        x = super().forward(x)
        x = sigmoid(x)
        return x


class SoftmaxMixin(OutFeaturesMixin):
    """
    Modifies the base model by adding a fully connected layer the same size as the number of possible classes, as well
    as a softmax activation function after the output of extended model.
    The mixin must be inherited before the `torch.nn.Module` to be extended in order to remove the additional
    `out_features` parameter used in the `OutFeaturesMixin` constructor.
    """
    def forward(self, x):
        x = super().forward(x)
        x = F.softmax(x, dim=-1)
        return x


class LogSoftmaxMixin(OutFeaturesMixin):
    """
    Modifies the base model by adding a fully connected layer the same size as the number of possible classes, as well
    as a log-softmax activation function after the output of extended model.
    The mixin must be inherited before the `torch.nn.Module` to be extended in order to remove the additional
    `out_features` parameter used in the `OutFeaturesMixin` constructor.
    """
    def forward(self, x):
        x = super().forward(x)
        x = F.log_softmax(x, dim=-1)
        return x
