__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixins for adding additional features to DataSet objects. """


import torch

from random import choice, randint

from src.utils import str_to_bin_array


class TripletMixin:
    """
    Mixin class for loading triplets on __get_item__ for any Dataset. Must be used with a torch.Dataset subclass,
    as it assumes the existence of the `classes`, `class_to_idx` and `imgs` fields.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with class keys, for efficient online
        triplet generation.
        """
        super().__init__(*args, **kwargs)
        self.imgs_dict = {
            idx: [index for index, img in enumerate(self.imgs) if img[1] == idx]
            for cls, idx in self.class_to_idx.items()
        }

    def __get_random_item__(self, cls):
        """
        Get a random item from the specified class.
        :param cls: the class idx
        :type: int
        :return: an item tuple
        :type: tuple
        """
        class_image_indexes = self.imgs_dict[cls]
        return super().__getitem__(class_image_indexes[randint(0, len(class_image_indexes) - 1)])

    def __getitem__(self, index):
        """
        Return a triplet consisting of an anchor (the indexed item), a positive (a random example of a different class),
        and a negative (a random example of the same class).
        :param index: an item's index
        :type: int
        :return: a tuple with the anchor
        :type: tuple(torch.Tensor, list<torch.Tensor>, int)
        """
        positive_class = self.imgs[index][1]
        negative_classes = list(range(0, positive_class)) + list(range(positive_class + 1, len(self.classes)))
        anchor = super().__getitem__(index)
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
        Initialize de base Dataset class and create a image index dictionary with file names as keys and dataset indexes
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
        the dataset. This is used to determine a standard binary encoding length for all indexes.
        """
        super().__init__(*args, **kwargs)
        self.max_binary_digits = len(str_to_bin_array(len(self.classes)))

    def __get_binary_encoding__(self, index):
        return torch.tensor(str_to_bin_array(index, self.max_binary_digits))
