__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Adaptation of torchvision ResNext models to fit the different datasets. """


from overrides import overrides
from torch import nn
from torchvision.models import resnext50_32x4d

from vscvs.models.mixins import NormalizedMixin, SigmoidMixin, SoftmaxMixin, LogSoftmaxMixin, OutFeaturesMixin


class ResNextBase(nn.Module):
    """
    ResNext model wrapper for easy swapping between ResNext models.
    """
    def __init__(self, pretrained=False, progress=True, **kwargs):
        """
        :param pretrained: if True, uses a model pre-trained on ImageNet.
        :type: boolean
        :param progress: if True, displays a progress bar of the download to stderr
        :type: boolean
        :param kwargs: additional `resnext50_32x4d` keyword arguments
        :type: dict
        """
        super().__init__()
        self.base = resnext50_32x4d(pretrained=pretrained, progress=progress, **kwargs)

    @overrides
    def forward(self, x):
        return self.base(x)


class ResNext(OutFeaturesMixin, ResNextBase):
    pass


class ResNextNormalized(NormalizedMixin, ResNextBase):
    pass


class ResNextSigmoid(SigmoidMixin, ResNextBase):
    pass


class ResNextSoftmax(SoftmaxMixin, ResNextBase):
    pass


class ResNextLogSoftmax(LogSoftmaxMixin, ResNextBase):
    pass
