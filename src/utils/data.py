__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utility functions for handling data. """


import random
import torch

from ignite._utils import convert_tensor
from torch.utils.data import Subset

from src.utils.decorators import deprecated


def simple_split(data, split_proportion=.8):
    """
    Splits incoming data into two sets, simply slicing on the index corresponding to the given proportion.
    :param data: the dataset to be split
    :type: indexed object
    :param split_proportion: proportion of the data to be assigned to the fist split subset. As this function returns
    two subsets, this parameter must be strictly between 0.0 and 1.0
    :type: float
    :return: the two resulting datasets
    :type: tup<indexed obj, indexed obj>
    """
    assert 0. < split_proportion < 1.
    split_index = int(len(data) * split_proportion)
    return data[:split_index], data[split_index:]


def random_simple_split(data, split_proportion=.8):
    """
    Splits incoming data into two sets, randomly and with no overlapping. Returns the two resulting data objects along
    with two arrays containing the original indexes of each element.
    :param data: the data to be split
    :type: indexed obj
    :param split_proportion: proportion of the data to be assigned to the fist split subset. As this function returns
    two subsets, this parameter must be strictly between 0.0 and 1.0
    :type: float
    :return: the two resulting datasets and the original indexes lists
    :type: indexed obj, indexed obj, list<int>, list<int>
    """
    assert 0. < split_proportion < 1.
    indexes = list(range(len(data))) # all indexes in data
    random.shuffle(indexes)
    split_index = int(len(data) * split_proportion)
    return data[indexes[:split_index]], data[indexes[split_index:]], indexes[:split_index], indexes[split_index:]


def split(data, split_proportion=.8):
    """
    Splits incoming data into two sets, one for training and one for tests. Non-overlapping.
    :param data: the dataset to be split
    :type: indexed object
    :param split_proportion: proportion of the data to be assigned to the fist split subset. As this function returns
    two subsets, this parameter must be strictly between 0.0 and 1.0
    :type: float
    :return: the two resulting datasets
    :type: indexed object
    """
    assert 0. < split_proportion < 1.
    test_index = int(len(data) * split_proportion)
    return Subset(data, range(test_index)), Subset(data, range(test_index, len(data)))


@deprecated
def dataset_split(dataset, train_test_split, train_validation_split):
    """
    Split a dataset into training. validation and test datasets, given the provided split proportions.
    Note: [len() has O(1) complexity](https://wiki.python.org/moin/TimeComplexity)
    :param dataset: the dataset to be split.
    :type: torch.utils.data.Dataset
    :param train_test_split: proportion of the dataset that will be used for training. The remaining
    data will be used as the test set.
    :type: float
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    :return: the the resulting Datasets
    :type: torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
    """
    n_test = int((1 - train_test_split) * len(dataset))
    n_train = int(train_validation_split * (len(dataset) - n_test))
    n_validation = len(dataset) - n_train - n_test
    return torch.utils.data.random_split(dataset, (n_train, n_validation, n_test))


def dataset_split_successive(dataset, *split_proportions):
    """
    Split a dataset successively into multiple subsets, given the provided split proportions. The first subset is
    taken from the original dataset using the first proportion. The following subsets are taken using the same logic,
    :param dataset: the dataset to be split.
    :type: torch.utils.data.Dataset
    :param split_proportions: any number of split proportions.
    :type: float $\in [0, 1]$
    :return: the the resulting Datasets
    :type: list<torch.utils.data.Dataset> same length as `split_proportions`
    """
    subset_lengths = []
    remaining_n = len(dataset)
    for split_proportion in split_proportions:
        subset_length = remaining_n * split_proportion
        subset_lengths.append(subset_length)
        remaining_n = max(0, remaining_n - subset_length)
    return torch.utils.data.random_split(dataset, subset_lengths)


def prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare batch for training: pass to a device with options. Assumes data and labels are the first
    two parameters of each sample.
    :param batch: data to be sent to device.
    :type: list
    :param device: device type specification
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    """
    x, y, *_ = batch # unpack extra parameters into `_`
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def prepare_batch_gan(batch, device=None, non_blocking=False):
    """
    Prepare batch for GAN training: pass to a device with options. Assumes the shape returned
    by the SketchyMixedBatches Dataset.
    :param batch: data to be sent to device.
    :type: list
    :param device: device type specification
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    """
    photos, sketches, classes = batch
    return (
        convert_tensor(photos, device=device, non_blocking=non_blocking),
        [convert_tensor(sketch, device=device, non_blocking=non_blocking) for sketch in sketches],
        convert_tensor(classes, device=device, non_blocking=non_blocking)
    )


def output_transform_gan(output):
    # `output` variable is returned by above `process_function`
    y_pred = output['y_pred']
    y = output['y_true']
    return y_pred, y  # output format is according to `Accuracy` docs