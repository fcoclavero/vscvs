__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Utility functions for handling data. """


import random

from collections import defaultdict

import torch

from torch.utils.data import Subset

from vscvs.decorators import deprecated


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
    :param split_proportions: any number of split proportions. Must be a `float` $\in [0, 1]$.
    :type: float
    :return: the the resulting Datasets, with lengths matching `split_proportions * len(dataset)`.
    :type: List[torch.utils.data.Dataset]
    """
    subset_lengths = []
    remaining_n = len(dataset)
    for split_proportion in split_proportions:
        subset_length = int(remaining_n * split_proportion)
        subset_lengths.append(subset_length)
        remaining_n = max(0, remaining_n - subset_length)
    subset_lengths.append(remaining_n)
    return torch.utils.data.random_split(dataset, [int(subset_length) for subset_length in subset_lengths])


def images_by_class(dataset):
    """
    Creates an image dictionary with class keys, useful for efficient online pair or triplet generation.
    :param dataset: the torch dataset from which the dictionary will be generated
    :type: torch.utils.data.Dataset
    :return: a dictionary with `dataset` classes as keys and a list of `dataset` indices as values
    :type: Dict
    """
    images_dict = defaultdict(list)  # if a new key used, it will be initialized with an empty list by default
    for image_index, image_class in enumerate(dataset.targets):  # `dataset.target` contains the class of each image
        images_dict[image_class].append(image_index)
    return images_dict


def random_simple_split(data, split_proportion=0.8):
    """
    Splits incoming data into two sets, randomly and with no overlapping. Returns the two resulting data objects along
    with two arrays containing the original indices of each element.
    :param data: the data to be split
    :type: SupportsIndex
    :param split_proportion: proportion of the data to be assigned to the fist split subset. As this function returns
    two subsets, this parameter must be strictly between 0.0 and 1.0
    :type: float
    :return: the two resulting datasets and the original index lists
    :type: SupportsIndex, SupportsIndex, List[int], List[int]
    """
    assert 0.0 < split_proportion < 1.0
    indices = list(range(len(data)))  # all indices in data
    random.shuffle(indices)
    split_index = int(len(data) * split_proportion)
    return data[indices[:split_index]], data[indices[split_index:]], indices[:split_index], indices[split_index:]


def simple_split(data, split_proportion=0.8):
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
    assert 0.0 < split_proportion < 1.0
    split_index = int(len(data) * split_proportion)
    return data[:split_index], data[split_index:]


def split(data, split_proportion=0.8):
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
    assert 0.0 < split_proportion < 1.0
    test_index = int(len(data) * split_proportion)
    return Subset(data, range(test_index)), Subset(data, range(test_index, len(data)))
