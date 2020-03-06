__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Adaptation of torchvision ResNet models to fit the different datasets. """


from torch import nn
from torchvision.models import resnet50

from src.models.convolutional.mixins import ClassificationMixin, OutFeaturesMixin


class ResNetBase(nn.Module):
    """
    ResNet model wrapper for easy swapping between ResNet models.
    """
    def __init__(self, *args, pretrained=False, progress=True, **kwargs):
        """
        Initialize model.
        :param args: additional arguments
        :type: tuple
        :param pretrained: if True, uses a model pre-trained on ImageNet.
        :type: boolean
        :param progress: if True, displays a progress bar of the download to stderr
        :type: boolean
        :param kwargs: additional keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.base = resnet50(pretrained=pretrained, progress=progress, **kwargs)

    def forward(self, x):
        return self.base(x)


class ResNetClassification(ClassificationMixin, ResNetBase):
    pass


class ResNet(OutFeaturesMixin, ResNetBase):
    pass
