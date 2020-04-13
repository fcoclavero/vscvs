__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Datasets for managing multimodal data loading. """


from numpy.random import choice
from torch.utils.data import Dataset

from vscvs.utils.data import images_by_class


class MultimodalDataset(Dataset):
    """
    Dataset class for loading elements from multiple datasets on `__getitem__`. The first Dataset is used as base: the
    length of the multimodal Dataset corresponds to that of the base dataset, and the first item in the tuples returned
    on `__getitem__` corresponds to item with the requested index in the base dataset.
    """
    def __init__(self, base_dataset, *args, **kwargs):
        """
        Dataset constructor. Creates an image dictionary with class keys, for efficient online pair generation.
        :param base_dataset: base of the siamese Dataset. The length of the multimodal Dataset corresponds to this
        dataset's length, and the first item in the pairs returned on `__getitem__` corresponds to item with the
        requested index in this dataset.
        :type: torch.utils.data.Dataset
        :param args: additional arguments
        :type: tuple
        :param kwargs: additional keyword arguments
        :type: dict
        """
        self.base_dataset = base_dataset
        super().__init__()

    def __len__(self):
        """
        Override the `__len__` property of the Dataset object to match the base Dataset.
        :return: the length of the drawable items in the siamese Dataset.
        :type: int
        """
        return len(self.base_dataset)


class SiameseDataset(MultimodalDataset):
    """
    Dataset class for loading random online pairs on `__getitem__` for the given pair of Datasets. The first Dataset is
    used as base: the length of the siamese Dataset corresponds to that of the first dataset, and the first item in the
    pairs returned on `__getitem__` corresponds to item with the requested index in the base dataset (plus a random
    item from the second dataset).
    """
    def __init__(self, base_dataset, paired_dataset, *args, positive_pair_proportion=0.5, **kwargs):
        """
        Dataset constructor. Creates an image dictionary with class keys, for efficient online pair generation.
        :param base_dataset: base of the siamese Dataset. The length of the siamese Dataset corresponds to this
        dataset's length, and the first item in the pairs returned on `__getitem__` corresponds to item with the
        requested index in this dataset.
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__getitem__`.
        It must have the same classes as `base_dataset`.
        :type: torch.utils.data.Dataset
        :param args: additional arguments
        :type: tuple
        :param positive_pair_proportion: proportion of pairs that will be positive (same class).
        :type: float
        :param kwargs: additional keyword arguments
        :type: dict
        """
        self.paired_dataset = paired_dataset
        self.target_probabilities = [positive_pair_proportion, 1 - positive_pair_proportion] # siamese target value prob
        self.paired_image_dict = images_by_class(paired_dataset)
        super().__init__(base_dataset, *args, **kwargs)

    def _get_pair(self, first_item_class):
        """
        Get a siamese pair from the paired dataset, which will be randomly positive (same class) or negative.
        :param first_item_class: the idx of the first siamese pair element's class.
        :type: int
        :return: an item tuple
        :type: tuple
        """
        target = choice([0, 1], p=self.target_probabilities) # if `target==0` ...
        negative_classes = self._negative_classes(first_item_class)
        paired_item_cls = choice(negative_classes) if target else first_item_class # ... generate a positive pair
        return self._get_random_paired_item(paired_item_cls)

    def _get_random_paired_item(self, cls):
        """
        Get a random item from the paired dataset belonging to the given class.
        :param cls: the idx of the class to which the returned item must belong
        :type: int
        :return: a random item belonging to `cls`
        :type: torch.Tensor
        """
        item_index = choice(self.paired_image_dict[cls]) # random index from list with all indices that belong to `cls`
        return self.paired_dataset[item_index]

    def _negative_classes(self, cls):
        """
        Return a list with all dataset classes that are not `cls`.
        :param cls: the positive class idx
        :type: int
        :return: a list with all negative classes idx
        :type: list<int>
        """
        classes = list(self.paired_image_dict.keys())
        classes.remove(cls)
        return classes

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method to return siamese pairs: the item in the base Dataset at `index` along
        with another random item from the paired Dataset. The pair will be randomly positive (same class) or negative.
        :param index: an item's index
        :type: int
        :return: a 2-tuple with the item corresponding to `index` in the base Dataset, along with another random item
        from the paired Dataset.
        :type: tuple<tuple, tuple>
        """
        item = self.base_dataset[index]
        item_class = item[1]
        return item, self._get_pair(item_class)


class TripletDataset(MultimodalDataset):
    """
    Dataset class for loading random online triplets on `__getitem__` for the given pair of Datasets. The first Dataset
    is used as base: the length of the triplet Dataset corresponds to that of the first dataset, and the first item in
    the triplets returned on `__getitem__` corresponds to the item with the requested index in the base dataset (the
    triplet anchor). Other two random items from the second dataset are drawn, one with the same class as the anchor
    (the positive) and one of a different random class (the negative).
    """
    def __init__(self, base_dataset, paired_dataset, *args, **kwargs):
        """
        Dataset constructor. Creates an image dictionary with class keys, for efficient online pair generation.
        :param base_dataset: base of the triplet Dataset. The length of the siamese Dataset corresponds to this
        dataset's length, and anchor items
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__getitem__`.
        It must have the same classes as `base_dataset`.
        :type: torch.utils.data.Dataset
        :param args: additional arguments
        :type: tuple
        :param kwargs: additional keyword arguments
        :type: dict
        """
        self.paired_dataset = paired_dataset
        self.paired_image_dict = images_by_class(paired_dataset)
        super().__init__(base_dataset, *args, **kwargs)

    def _get_random_negative(self, anchor_item_class):
        """
        Get a random negative triplet element (an element in the paired dataset that does not belong to the same class
        as the anchor element).
        :param anchor_item_class: the idx of the anchor element's class.
        :type: int
        :return: an item tuple
        :type: tuple
        """
        negative_classes = self._negative_classes(anchor_item_class)
        paired_item_cls = choice(negative_classes)
        return self._get_random_paired_item(paired_item_cls)

    def _get_random_paired_item(self, cls):
        """
        Get a random item from the paired dataset belonging to the given class.
        :param cls: the idx of the class to which the returned item must belong
        :type: int
        :return: a random item belonging to `cls`
        :type: torch.Tensor
        """
        item_index = choice(self.paired_image_dict[cls]) # random index from list with all indices that belong to `cls`
        return self.paired_dataset[item_index]

    def _negative_classes(self, cls):
        """
        Return a list with all dataset classes that are not `cls`.
        :param cls: the positive class idx
        :type: int
        :return: a list with all negative classes idx
        :type: list<int>
        """
        classes = list(self.paired_image_dict.keys())
        classes.remove(cls)
        return classes

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method to return triplets: the item in the base Dataset at `index` (the
        anchor), a random item from the paired Dataset belonging to the same class as the anchor (the positive), and a
        random item from the paired dataset belonging to a random class different to that of the anchor (the negative).
        :param index: an item's index
        :type: int
        :return: a 3-tuple with an anchor from the base class, and a positive and negative from the paired dataset.
        :type: tuple<tuple, tuple, tuple>
        """
        item = self.base_dataset[index]
        item_class = item[1]
        return item, self._get_random_paired_item(item_class), self._get_random_negative(item_class)
