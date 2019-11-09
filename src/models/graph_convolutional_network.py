__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Graph convolutional network PyTorch modules. """


import torch

import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, batch_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(batch_size, batch_size)
        self.conv2 = GCNConv(batch_size, batch_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class ClassificationGCN(torch.nn.Module):
    def __init__(self, batch_size, num_classes):
        super(ClassificationGCN, self).__init__()
        self.conv1 = GCNConv(batch_size, batch_size)
        self.conv2 = GCNConv(batch_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
