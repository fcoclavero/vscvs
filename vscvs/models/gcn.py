__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Graph convolutional network PyTorch modules. """


import torch

import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from vscvs.models.hog import HOG
from vscvs.utils.data import prepare_batch_graph


class GCNClassification(torch.nn.Module):
    """
    GCN node classifier.
    """
    def __init__(self, num_classes, in_channels):
        """
        :param num_classes: number of possible node classes. The module output will be a vector with a length of
        `num_classes`, where the value in each position corresponds to the probability of a node belonging to
        each class.
        :type: int
        :param in_channels: length of node feature vectors.
        :type: int
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, num_classes)

    def forward(self, batch_graph):
        """
        Perform a forward pass through the module.
        :param batch_graph: a single graph, containing node feature vectors, connectivity matrix and edge weights.
        :type: torch_geometric.Data
        :return: tensor containing class probabilities for every node in the graph
        :type: torch.Tensor with shape <batch_size, num_classes>
        """
        x, edge_index, edge_weight = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.softmax(x, dim=1)


class HOGGCN(torch.nn.Module):
    """
    Image classifier that extracts HOG feature vectors for each image, and classifies using a GCN over clique graphs
    constructed from each batch, where edge weights correspond to class label word vector distances.
    """
    def __init__(self, classes_dataframe, in_dimension=256, in_channels=3, cell_size=8, bins=9, signed_gradients=False,
                 processes=None):
        """
        :param classes_dataframe: dataframe containing all possible class names and their word vectors
        :type: pandas.Dataframe
        :param in_dimension: input image dimensions (assuming square images).
        :type: int
        :param in_channels: number of input image color channels.
        :type: int
        :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
        calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
        :type: int
        :param bins: number of bins for the histogram of each cell.
        :type: int
        :param signed_gradients: gradients are represented using its angle and magnitude. Angles can be expressed
        using values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the
        gradient “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has
        been shown that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
        :type: boolean
        :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
        `os.cpu_count()` will be used.
        :type: int or None
        """
        super().__init__()
        self.classes_dataframe = classes_dataframe
        self.processes = processes
        self.hog = HOG(in_channels=in_channels, cell_size=cell_size, bins=bins, signed_gradients=signed_gradients)
        self.classification_gcn = GCNClassification(len(classes_dataframe), self.hog.descriptor_length(in_dimension))

    def _batch_graph(self, batch, device=None, non_blocking=False):
        """
        Prepare batch for training: pass to a device with options. Assumes data and labels are the first
        two parameters of each sample.
        :param batch: data to be sent to device.
        :type: list
        :param device: device type specification
        :type: str of torch.device (optional) (default: None)
        :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
        :type: bool (optional)
        """
        return prepare_batch_graph(batch, self.classes_dataframe, device, non_blocking, self.processes)

    def forward(self, image_batch):
        """
        Perform a forward pass through the module.
        :param image_batch: batch of input images.
        :type: torch.Tensor with shape <batch_size, in_channels, in_dimension, in_dimension>
        :return: tensor containing class probabilities for every node in the graph
        :type: torch.Tensor with shape <batch_size,
        """
        x, y, *_ = image_batch  # unpack extra parameters into `_`
        embeddings = self.hog(x)
        return self.classification_gcn(self._batch_graph( (embeddings, y) ))
