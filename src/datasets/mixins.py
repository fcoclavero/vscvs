__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixins for adding additional features to DataSet objects. """


import torch

from random import choice, randint

from src.utils import str_to_bin_array
from src.utils.data import images_by_class


class SiameseMixin:
    """
    Mixin class for loading random pairs on `__get_item__` for any Dataset. Must be used with a torch.Dataset subclass,
    as it assumes the existence of the `classes`, `class_to_idx` and `imgs` fields.
    """
    def __get_random_item__(self):
        """
        Get a random item from the Dataset.
        :return: an item tuple
        :type: tuple
        """
        return super().__getitem__(randint(0, len(self) - 1))

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method, returning each requested item along with another random item from
        the Dataset.
        :param index: an item's index
        :type: int
        :return: a 2-tuple with the item corresponding to `index`, along with another random item.
        :type: tuple
        """
        return super().__getitem__(index), self.__get_random_item__()


class TripletMixin:
    """
    Mixin class for loading triplets on `__get_item__` for any Dataset. Must be used with a torch.Dataset subclass,
    as it assumes the existence of the `classes`, `class_to_idx` and `imgs` fields.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with class keys, for efficient online
        triplet generation.
        """
        super().__init__(*args, **kwargs)
        self.image_dict = images_by_class(self)

    def __get_random_item__(self, cls):
        """
        Get a random item from the specified class.
        :param cls: the class idx
        :type: int
        :return: an item tuple
        :type: tuple
        """
        return super().__getitem__(choice(self.image_dict[cls]))

    def __getitem__(self, index):
        """
        Return a triplet consisting of an anchor (the indexed item), a positive (a random example of a different class),
        and a negative (a random example of the same class).
        :param index: an item's index
        :type: int
        :return: a 3-tuple with the anchor, positive and negative
        :type: tuple
        """
        anchor = super().__getitem__(index)
        positive_class = self.targets[index]
        negative_classes = list(range(0, positive_class)) + list(range(positive_class + 1, len(self.classes)))
        positive = self.__get_random_item__(positive_class)
        negative = self.__get_random_item__(choice(negative_classes))
        return anchor, positive, negative


class FilenameIndexedMixin:
    """
    Mixin class for getting dataset items from a file name. It is intended to be used for retrieval tasks.
    The mixin must be used with a torch.Dataset subclass, as it assumes the existence of the `imgs` field and a
    `__getitem__` with the default Dataset indexation.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with file names as keys and dataset indices
        as values for efficient retrieval after initialization.
        """
        super().__init__(*args, **kwargs)
        self.imgs_dict = {tup[0]: i for i, tup in enumerate(self.imgs)}

    def getitem_by_filename(self, filename):
        """
        Get and item based on it's filename by getting the item's real index using the `imgs_dict` defined in the
        FilenameIndexedMixin mixin and then using the index to retrieve the item using the default `__getitem__`.
        :param filename: the filename of the item to be retrieved. Uses full paths.
        :type: str
        :return: the corresponding Dataset item.
        :type: same as the mixed Dataset's `__getitem__`
        """
        return self[self.imgs_dict[filename]]


class BinaryEncodingMixin:
    """
    Mixin class for adding unique binary encoding descriptors for each element in the dataset.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and compute the length (in digits) of the binary form of the largest index in
        the dataset. This is used to determine a standard binary encoding length for all indices.
        """
        super().__init__(*args, **kwargs)
        self.max_binary_digits = len(str_to_bin_array(len(self.targets)))

    def __get_binary_encoding__(self, index):
        bin_arr = torch.tensor(str_to_bin_array(index, self.max_binary_digits))
        return bin_arr

    def __getitem__(self, item):
        return self.__get_binary_encoding__(item), self.targets[item]


class OneHotEncodingMixin:
    """
    Mixin class for adding unique one-hot encoding descriptors for each element in the dataset.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a tensor with all one hot encodings.
        """
        super().__init__(*args, **kwargs)
        self.encodings = torch.eye(len(self.targets))

    def __get_one_hot_encoding__(self, index):
        return self.encodings[index]

    def __getitem__(self, item):
        return self.__get_one_hot_encoding__(item), self.targets[item]
