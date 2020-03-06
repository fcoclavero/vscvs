__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utility model for adapting a convolutional model for classification. """


from torch import nn

import torch.nn.functional as F

from src.utils import get_out_features_from_model


class Classification(nn.Module):
    """
    Modifies the base model by adding a fully connected layer the same size as the number of possible classes, as well
    as using a softmax function on the output of the new model.
    """
    def __init__(self, base_model, out_features):
        """
        Initialize model.
        :param base_model: the model to be adapted for classification.
        :type: torch.nn.Module
        :param out_features: number of output features.
        :type: int
        """
        super().__init__()
        self.base_model = base_model
        self.fully_connected = nn.Linear(get_out_features_from_model(base_model), out_features)

    def forward(self, x):
        x = F.relu(self.base_model(x))
        x = self.fully_connected(x)
        x = F.log_softmax(x, dim=-1)
        return x
