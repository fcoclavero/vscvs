__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Datasets for managing multimodal data loading. """


from numpy.random import choice
from torch.utils.data import Dataset

from src.utils.data import images_by_class


class SiameseDataset(Dataset):
    """
    Dataset class for loading random online pairs on `__get_item__` for the given pair of Datasets. The first Dataset is
    used as base: the length of the siamese Dataset corresponds to that of the first dataset, and the first item in the
    pairs returned on `__get_item__` corresponds to item with the requested index in the base dataset (plus a random
    item from the second dataset).
    """
    def __init__(self, base_dataset, paired_dataset, positive_pair_proportion=0.5):
        """
        Dataset constructor. Creates an image dictionary with class keys, for efficient online pair generation.
        :param base_dataset: base of the siamese Dataset. The length of the siamese Dataset corresponds to this
        dataset's length, and the first item in the pairs returned on `__get_item__` corresponds to item with the
        requested index in this dataset.
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__get_item__`.
        It must have the same classes as `base_dataset`.
        :type: torch.utils.data.Dataset
        :param positive_pair_proportion: proportion of pairs that will be positive (same class).
        :type: float
        """
        self.base_dataset = base_dataset
        self.paired_dataset = paired_dataset
        self.target_probabilities = [positive_pair_proportion, 1 - positive_pair_proportion] # siamese target value prob
        self.target_image_dict = images_by_class(paired_dataset)

    def __len__(self):
        """
        Override the `__len__` property of the Dataset object to match the base Dataset.
        :return: the length of the drawable items in the siamese Dataset.
        :type: int
        """
        return len(self.base_dataset)

    def __get_random_paired_item__(self, cls):
        """
        Get a random item from the paired dataset belonging to the given class.
        :param cls: the idx of the class to which the returned item must belong
        :type: int
        :return: a random item belonging to `cls`
        :type: torch.Tensor
        """
        item_index = choice(self.target_image_dict[cls]) # random index from list with all indices that belong to `cls`
        return self.paired_dataset[item_index]

    def __get_pair__(self, first_item_class):
        """
        Get a siamese pair from the paired dataset, which will be randomly positive (same class) or negative.
        :param first_item_class: the idx of the first siamese pair element's class.
        :type: int
        :return: an item tuple
        :type: tuple
        """
        target = choice([0, 1], p=self.target_probabilities) # if `target==0` ...
        negative_classes = self.__negative_classes__(first_item_class)
        paired_item_cls = choice(negative_classes) if target else first_item_class # ... generate a positive pair
        return self.__get_random_paired_item__(paired_item_cls)

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method, returning the item in the base Dataset at `index` along with another
        random item from the paired Dataset. The pair will be randomly positive (same class) or negative.
        :param index: an item's index
        :type: int
        :return: a 2-tuple with the item corresponding to `index` in the base Dataset, along with another random item
        from the paired Dataset.
        :type: tuple
        """
        item = self.base_dataset[index]
        item_class = item[1]
        return item, self.__get_pair__(item_class)

    def __negative_classes__(self, cls):
        """
        Return a list with all dataset classes that are not `cls`.
        :param cls: the positive class idx
        :type: int
        :return: a list with all negative classes idx
        :type: list<int>
        """
        classes = list(self.target_image_dict.keys())
        classes.remove(cls)
        return classes
