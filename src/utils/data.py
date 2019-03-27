import torch


def split(data, split_proportion = 0.8):
    """
    Splits incoming data into two sets, one for training and one for tests.
    Current implementation just slices on the index corresponding to the given proportion. This could be changed
    to a random, class balanced version.
    :param data: the dataset to be split
    :type: indexed object
    :param split_proportion:
    :return: the two resulting datasets
    :type: indexed object
    """
    test_index = int(len(data) * split_proportion)
    return data[:test_index], data[test_index:]


def dataset_split(dataset, train_test_split, train_validation_split):
    """
    Split a dataset into training. validation and test datasets, given the provided split proportions.
    Note: [len() has O(1) complexity](https://wiki.python.org/moin/TimeComplexity)
    :param dataset: the dataset to be split.
    :type: torch.utils.data.Dataset
    :param train_test_split: proportion of the dataset that will be used for training. The remaining data will be used
    as the test set.
    :type: float
    :param train_validation_split: proportion of the training set that will be used for actual training. The remaining
    data will be used as the validation set.
    :type: float
    :return: the the resulting Datasets
    :type: torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
    """
    n_test = int((1 - train_test_split) * len(dataset))
    n_train = int(train_validation_split * (len(dataset) - n_test))
    n_validation = len(dataset) - n_train - n_test
    return torch.utils.data.random_split(dataset, (n_train, n_validation, n_test))