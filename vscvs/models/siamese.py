__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Siamese network model definition. """


import torch.nn as nn


class SiameseNetwork(nn.Module):
    """
    Pytorch module for a siamese network.
    Siamese networks produce similar embeddings for similar inputs by training using a contrastive loss function that
    has penalizes similar embeddings for two images of different classes, as well as dissimilar embeddings for two
    images of the same class.
    """
    def __init__(self, embedding_network_0, embedding_network_1):
        """
        :param embedding_network_0: the network that will encode the first element of each sample pair.
        :type: torch.nn.module
        :param embedding_network_1: the network that will encode the first element of each sample pair.
        :type: torch.nn.module
        """
        super().__init__()
        self.embedding_network_0 = embedding_network_0
        self.embedding_network_1 = embedding_network_1

    def forward(self, input_0, input_1):
        """
        Perform a forward pass on the network, computing the embeddings for both inputs.
        :param input_0: the first network input
        :type: torch.Tensor with a size compatible with `embedding_network_1`
        :param input_1: the second network input
        :type: torch.Tensor with a size compatible with `embedding_network_2`
        :return: the embeddings for both inputs
        :type: torch.Tensor, torch.Tensor
        """
        embedding_0 = self.embedding_network_0(input_0)
        embedding_1 = self.embedding_network_1(input_1)
        return embedding_0, embedding_1


class SiameseNetworkShared(nn.Module):
    """
    Pytorch module for a siamese network with shared weights and architecture.
    Siamese networks produce similar embeddings for similar inputs by training using a contrastive loss function that
    has penalizes similar embeddings for two images of different classes, as well as dissimilar embeddings for two
    images of the same class.
    """
    def __init__(self, embedding_network):
        """
        :param embedding_network: the network to be used in the siamese training. It must accept network inputs and
        produce network outputs.
        :type: torch.nn.module
        """
        super().__init__()
        self.embedding_network = embedding_network

    def forward(self, input_0, input_1):
        """
        Perform a forward pass on the network, computing the embeddings for both inputs.
        :param input_0: the first network input
        :type: torch.Tensor with a size compatible with `embedding_network`
        :param input_1: the second network input
        :type: torch.Tensor with a size compatible with `embedding_network`
        :return: the embeddings for both inputs
        :type: torch.Tensor, torch.Tensor
        """
        embedding_0 = self.embedding_network(input_0)
        embedding_1 = self.embedding_network(input_1)
        return embedding_0, embedding_1
