__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Triplet network model definition. """


import torch.nn as nn


class TripletNetwork(nn.Module):
    """
    Pytorch module for a triplet network.
    Triplet networks produce similar embeddings for similar inputs by training using a triplet loss function that
    receives three input examples, an anchor, a positive (same class as anchor) and a negative (different class thant
    anchor's). It penalizes similar embeddings for the anchor and the negative, as well as dissimilar embeddings for the
    anchor and the positive.
    Positive and negative elements in each triplet are assumed to be of the same mode, and thus their embedding network
    will share weights.
    """
    def __init__(self, anchor_embedding_network, positive_negative_embedding_network):
        """
        Model constructor.
        :param anchor_embedding_network: the network that will encode the anchor element of each triplet.
        :type: torch.nn.module
        :param positive_negative_embedding_network: the network that will encode the positive and negative elements.
        :type: torch.nn.module
        """
        super().__init__()
        self.anchor_embedding_network = anchor_embedding_network
        self.positive_negative_embedding_network = positive_negative_embedding_network

    def forward(self, anchor, positive, negative):
        """
        Perform a forward pass on the network, computing the embeddings for the triplet.
        :param anchor: the first network input, with respect to which the rest of the inputs will be compared.
        :type: torch.Tensor with a size compatible with `anchor_embedding_network`
        :param positive: the second network input, an element of the same class as `anchor`
        :type: torch.Tensor with a size compatible with `positive_negative_embedding_network`
        :param negative: the third network input, an element with a class different to `anchor`
        :type: torch.Tensor with a size compatible with `positive_negative_embedding_network`
        :return: the embeddings for all triplet elements
        :type: torch.Tensor, torch.Tensor
        """
        anchor_embedding = self.anchor_embedding_network(anchor)
        positive_embedding = self.positive_negative_embedding_network(positive)
        negative_embedding = self.positive_negative_embedding_network(negative)
        return anchor_embedding, positive_embedding, negative_embedding
