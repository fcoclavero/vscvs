__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Adaptation of torchvision ResNet models to fit the different datasets. """


from overrides import overrides
from torch import nn
from torchvision.models import resnet50

from vscvs.models.mixins import NormalizedMixin, SigmoidMixin, SoftmaxMixin, LogSoftmaxMixin, OutFeaturesMixin


class ResNetBase(nn.Module):
    """
    ResNet model wrapper for easy swapping between ResNet models.
    """
    def __init__(self, pretrained=False, progress=True, **kwargs):
        """
        :param pretrained: if True, uses a model pre-trained on ImageNet.
        :type: bool
        :param progress: if True, displays a progress bar of the download to stderr
        :type: bool
        :param kwargs: additional `resnet50` keyword arguments
        :type: Dict
        """
        super().__init__()
        self.base = resnet50(pretrained=pretrained, progress=progress, **kwargs)

    @overrides
    def forward(self, x):
        return self.base(x)


class ResNet(OutFeaturesMixin, ResNetBase):
    pass


class ResNetNormalized(NormalizedMixin, ResNetBase):
    pass


class ResNetSigmoid(SigmoidMixin, ResNetBase):
    pass


class ResNetSoftmax(SoftmaxMixin, ResNetBase):
    pass


class ResNetLogSoftmax(LogSoftmaxMixin, ResNetBase):
    pass
