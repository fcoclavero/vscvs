__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Triplet network model definition. """


import torch.nn as nn
import torch.nn.functional as F


class TripletNetwork(nn.Module):
    def __init__(self, embedding_network):
        super().__init__()
        self.embedding_network = embedding_network

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.embedding_network(anchor)
        positive_embedding = self.embedding_network(positive)
        negative_embedding = self.embedding_network(negative)
        distance_to_positive = F.pairwise_distance(anchor_embedding, positive_embedding, 2)
        distance_to_negative = F.pairwise_distance(anchor_embedding, negative_embedding, 2)
        return anchor_embedding, positive_embedding, negative_embedding, distance_to_positive, distance_to_negative