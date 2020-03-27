__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixin for adapting a convolutional model for classification. """


import torch.nn.functional as F

from torch import nn
from torch import sigmoid

from src.utils import get_out_features_from_model


class OutFeaturesMixin:
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