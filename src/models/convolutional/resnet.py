__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Adaptation of torchvision ResNet models to fit the different datasets. """


import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50


class ResNet(nn.Module):
    """
    Torchvision `resnet50` model with an additional fully connected layer to match the desired output features.
    """
    def __init__(self, out_features=None, pretrained=False, progress=True, **kwargs):
        """
        Initialize model.
        :param out_features: number of output features. If `None`, the base `resnext50_32x4d` model will be used with
        no modifications, having an output feature number of 1000.
        :type: int or None
        :param pretrained: if True, uses a model pre-trained on ImageNet.
        :type: boolean
        :param progress: if True, displays a progress bar of the download to stderr"
        :type: boolean
        :param kwargs: additional keyword arguments
        :type: dict
        """
        super().__init__()
        self.resnet_base = resnet50(pretrained=pretrained, progress=progress, **kwargs) # 1000
        self.fully_connected = nn.Linear(1000, out_features)  # out_features

    def forward(self, x):
        x = F.relu(self.resnet_base(x))
        return self.fully_connected(x)
