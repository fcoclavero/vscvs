__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Batch preparation functions. """


from itertools import repeat

import torch

from ignite.utils import convert_tensor
from torch.multiprocessing import Pool
from torch_geometric.data import Data


def batch_clique_graph(batch, classes_dataframe, processes=None):
    """
    Creates a graph Data object from an image batch, to use with a semi-supervised graph learning model. The created
    graph connects all batch elements with each other (clique graph) and graph vertex weights correspond to word
    vector distances of class labels. Assumes data and labels are the first two parameters of each sample.
    :param batch: data to be sent to device.
    :type: List
    :param classes_dataframe: dataframe containing class names and their word vectors.
    :type: pandas.Dataframe
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int
    :return: the batch clique graph.
    :type: torch_geometric.data.Data
    """
    x, y, *_ = batch  # unpack extra parameters into `_`
    edge_index = torch.stack(
        [  # create the binary adjacency matrix for the clique graph
            torch.arange(x.shape[0]).repeat_interleave(x.shape[0]),  # each index repeated num_edges times
            torch.arange(x.shape[0]).repeat(x.shape[0]),
        ]
    )  # the index range repeated num_edges times
    with Pool(processes=processes) as pool:  # create edge weights from the word vector distances
        edge_classes = torch.stack([y.repeat_interleave(y.shape[0]), y.repeat(y.shape[0])]).t().contiguous()
        edge_attr = torch.stack(
            pool.starmap(wordvector_distance, zip(edge_classes, repeat(torch.tensor(classes_dataframe["distances"]))))
        )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare batch for training: pass to a device with options. Assumes data and labels are the first
    two parameters of each sample.
    :param batch: data to be sent to device.
    :type: List[torch.Tensor]
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :return: 2-tuple with batch elements and labels.
    :type: Tuple[torch.Tensor, torch.Tensor]
    """
    x, y, *_ = batch  # unpack extra parameters into `_`
    return tuple(convert_tensor(element, device=device, non_blocking=non_blocking) for element in [x, y])


def prepare_batch_graph(batch, classes_dataframe, device=None, non_blocking=False, processes=None):
    """
    Prepare batch for training: pass to a device with options. Assumes data and labels are the first
    two parameters of each sample.
    :param batch: data to be sent to device.
    :type: List[torch.Tensor]
    :param classes_dataframe: dataframe containing class names and their word vectors.
    :type: pandas.Dataframe
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int
    :return: the batch clique graph
    :type: torch_geometric.data.Data
    """
    graph = batch_clique_graph(batch, classes_dataframe, processes)
    graph.apply(lambda attr: convert_tensor(attr.float(), device=device, non_blocking=non_blocking), "x", "edge_attr")
    return graph


def prepare_batch_siamese(batch, device=None, non_blocking=False):
    """
    Prepare batch for siamese network training: pass to a device with options. Assumes the shape returned by a Dataset
    implementing the `SiameseMixin`.
    :param batch: data to be sent to device.
    :type: Tuple[List[torch.Tensor], List[torch.Tensor]]
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :return: 3-tuple with batches of siamese pairs and their target label.
    :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    images_0, images_1 = batch
    target = siamese_target(images_0, images_1)
    return tuple(convert_tensor(i, device=device, non_blocking=non_blocking) for i in [images_0, images_1, target])


def prepare_batch_multimodal(batch, device=None, non_blocking=False):
    """
    Prepare batch for multimodal network training: pass to a device with options. Assumes the shape returned by a
    `MultimodalDataset` subclass.
    :param batch: data to be sent to device.
    :type: List[List[torch.Tensor]]
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :return: tuple of length `n_modes` with multimodal batches.
    :type: List[List[torch.Tensor]]
    """
    return [prepare_batch(images, device, non_blocking) for images in batch]


def prepare_batch_multimodal_siamese(batch, device=None, non_blocking=False):
    """
    Prepare batch for siamese multimodal training: pass to a device with options. Assumes the shape returned by a
    `MultimodalEntitySiameseDataset` subclass.
    :param batch: data to be sent to device.
    :type: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :return: 3-tuple with multimodal batches for the siamese pairs and their siamese target tensor, which contains a
    `0` if a pair is similar (has the same class) or `1` otherwise.
    :type: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], torch.Tensor]
    """
    entities_0, entities_1 = batch
    assert torch.equal(entities_0[0][1], entities_0[1][1]) and torch.equal(entities_1[0][1], entities_1[1][1])
    target = siamese_target(entities_0, entities_1, lambda x: x[0][1])
    return (
        prepare_batch_multimodal(entities_0, device, non_blocking),
        prepare_batch_multimodal(entities_1, device, non_blocking),
        convert_tensor(target, device=device, non_blocking=non_blocking),
    )


def siamese_target(elements_0, elements_1, get_classes=lambda x: x[1]):
    """
    Creates the contrastive loss target vector for the given image pairs. A target of 0 is assigned when both images
    in a pair are *similar* (have the same class), 1 otherwise.
    :param elements_0: normal batch of the first elements of each siamese pair, which by default is a 2-tuple with the
    elements tensor and a labels tensor.
    :type: Tuple[torch.Tensor, torch.Tensor]
    :param elements_1: batch of the second elements of each siamese pair, which by default is a 2-tuple with the
    elements tensor and a labels tensor.
    :type: Tuple[torch.Tensor, torch.Tensor]
    :param get_classes: function that takes a standard batch from a siamese pair element and returns the labels tensor,
    In a standard dataset, the input for this function is a tuple where the first element is the data tensor and the
    second element is the labels tensor. In this default case the latter is returned.
    :type: Callable[[torch.Tensor], torch.Tensor]
    :return: tensor with the contrastive loss target.
    :type: torch.Tensor with shape `batch_size`
    """
    # noinspection PyUnresolvedReferences
    return (get_classes(elements_0) != get_classes(elements_1)).int()  # `torch.Tensor` type is inferred


def wordvector_distance(indices, class_wordvector_distances):
    """
    Get the distance of two class word vectors, given a pre-computed distance matrix. This can be used to determine
    edge weights between two batch graph nodes.
    :param indices: the indices of the two classes. In the graph structure, the first index corresponds to the origin
    vertex and the second index corresponds to the destination vertex. Tensor of shape `[2]`.
    :type: torch.Tensor
    :param class_wordvector_distances: pre-computed class word vector distances for all the classes in the dataset. The
    matrix is symmetrical.
    :type: pandas.Dataframe
    :return: the distances between the word vectors of the classes specified in `indices`.
    :type: float
    """
    return class_wordvector_distances[indices[0]][indices[1]]
