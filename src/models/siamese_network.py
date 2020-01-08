__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Siamese network model definition. """


import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """
    Pytorch module for a siamese network.
    Siamese networks produce similar embeddings for similar inputs by training using a contrastive loss function that
    has penalizes similar embeddings for two images of different classes, as well as dissimilar embeddings for two
    images of the same class.
    """
    def __init__(self, embedding_network):
        """
        Model constructor.
        :param embedding_network: the network to be used in the siamese training. It must accept network inputs and
        produce network outputs.
        :type: torch.nn.module
        """
        super().__init__()
        self.embedding_network = embedding_network

    def forward(self, input_1, input_2):
        """
        Perform a forward pass on the network, computing the embeddings for both inputs and the distance between them,
        as it will be needed for computing the loss.
        :param input_1: the first network input
        :type: torch.Tensor with a size compatible with `embedding_network`
        :param input_2: the second network input
        :type: torch.Tensor with a size compatible with `embedding_network`
        :return: the embeddings for both inputs and the distance between them
        :type: torch.Tensor, torch.Tensor, torch.Float
        """
        embedding_1 = self.embedding_network(input_1)
        embedding_2 = self.embedding_network(input_2)
        distance = F.pairwise_distance(embedding_1, embedding_2, 2)
        return embedding_1, embedding_2, distance
