__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Datasets for managing multimodal data loading. """


from random import randint
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    """
    Dataset class for loading random online pairs on `__get_item__` for the given pair of Datasets. The first Dataset is
    used as base: the length of the siamese Dataset corresponds to that of the first dataset, and the first item in the
    pairs returned on `__get_item__` corresponds to item with the requested index in the base dataset (plus a random
    item from the second dataset).
    """
    def __init__(self, base_dataset, paired_dataset):
        """
        Dataset constructor.
        :param base_dataset: base of the siamese Dataset. The length of the siamese Dataset corresponds to this
        dataset's length, and the first item in the pairs returned on `__get_item__` corresponds to item with the
        requested index in this dataset.
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__get_item__`.
        :type: torch.utils.data.Dataset
        """
        self.base_dataset = base_dataset
        self.paired_dataset = paired_dataset

    def __len__(self):
        """
        Override the `__len__` property of the Dataset object to match the base Dataset.
        :return: the length of the drawable items in the siamese Dataset.
        :type: int
        """
        return len(self.base_dataset)

    def __get_random_pair_item__(self):
        """
        Get a random item from the paired Dataset.
        :return: an item tuple
        :type: tuple
        """
        return self.paired_dataset[randint(0, len(self.paired_dataset) - 1)]

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method, returning the item in the base Dataset at `index` along with another
        random item from the paired Dataset.
        :param index: an item's index
        :type: int
        :return: a 2-tuple with the item corresponding to `index` in the base Dataset, along with another random item
        from the paired Dataset.
        :type: tuple
        """
        return self.base_dataset[index], self.__get_random_pair_item__()
