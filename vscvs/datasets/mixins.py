__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixins for adding additional features to DataSet objects. """


import os
import re
import pickle
import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from multipledispatch import dispatch
from numpy.random import choice
from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple

from vscvs.utils import get_cache_directory, str_to_bin_array


""" Type hinting utility mixin classes. """


class DatasetMixin:
    """
    Utility class that type hints `Dataset` methods that will be available to the mixins that inherit this class, as
    they are meant to be used in multiple inheritance with `torch.utils.data.Dataset`.
    """
    __add__: Callable[[Dataset], Dataset] # concatenates dataset parameter
    __getitem__: Callable[[int], tuple] # get the item tuple for the given index
    __len__: int # number of data points in the dataset'


class DatasetFolderMixin(DatasetMixin):
    """
    Utility class that type hints `DatasetFolder` methods that will be available to the mixins that inherit this class,
    as they are meant to be used in multiple inheritance with `torchvision.datasets.DatasetFolder`.
    """
    classes: List[str] # list of class names
    class_to_idx: Dict[str, int] # dictionary from class_name to class_index
    samples: List[Tuple[str, int]] # list of (sample_path, class_index) tuples
    targets: List[int] # list of class_index values for each sample


class ImageFolderMixin(DatasetFolderMixin):
    """
    Utility class that type hints `ImageFolder` methods that will be available to the mixins that inherit this class, as
    they are meant to be used in multiple inheritance with `torchvision.datasets.ImageFolder`.
    """
    imgs: List[Tuple[str, int]] # list of (image_path, class_index) tuples


class MultimodalDatasetMixin:
    """
    Utility class that type hints `MultimodalDataset` methods that will be available to the mixins that inherit this
    class, as they are meant to be used in multiple inheritance with `vscvs.datasets.multimodal.MultimodalDataset`.
    """
    base_dataset: DatasetFolder


""" Actual mixins. """


class BinaryEncodingMixin:
    """
    Mixin class for adding unique binary encoding descriptors for each element in the dataset.
    """
    targets: List[int]

    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and compute the length (in digits) of the binary form of the largest index in
        the dataset. This is used to determine a standard binary encoding length for all indices.
        :param args: super class arguments.
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.max_binary_digits = len(str_to_bin_array(len(self.targets)))

    def _get_binary_encoding(self, index):
        """
        Get the binary encoding of the given index.
        :param index: the index in decimal form.
        :type: int
        :return: a torch tensor with the binary form of the index.
        :type: torch.Tensor
        """
        bin_arr = torch.tensor(str_to_bin_array(index, self.max_binary_digits))
        return bin_arr

    def __getitem__(self, item):
        return self._get_binary_encoding(item), self.targets[item]


class ClassIndicesMixin(DatasetFolderMixin):
    """
    DatasetFolder mixin that creates a dictionary with dataset classes as keys and the corresponding dataset element
    indices as values. These can the be accessed via the `get_class_element_indices` method.
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: base Dataset class arguments
        :type: list
        :param kwargs: base Dataset class keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.class_element_indices_dict = defaultdict(list)  # if new key used, it will be instanced with an empty list
        for element_index, element_class in enumerate(self.targets):  # `self.target` contains the class of each element
            self.class_element_indices_dict[element_class].append(element_index)

    def get_class_element_indices(self, class_index):
        """
        Getter for all the elements of the requested class.
        :param class_index: the class index.
        :type: int
        :return: a list with all the elements
        """
        return self.class_element_indices_dict[class_index]


class FileNameIndexedMixin(ImageFolderMixin):
    """
    Mixin class for getting dataset items from a file name.
    To support indexing by index and name simultaneously, the `__getitem__` function was transformed to a single
    dispatch generic function using the `multiple-dispatch` module.
    Reference:
        1. [Python dispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch)
        2. [multiple-dispatch module](https://multiple-dispatch.readthedocs.io/en/latest/index.html)
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with file names as keys and dataset indices
        as values for efficient retrieval after initialization.
        :param args: super class arguments.
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self._imgs_dict = {self._get_image_name(i): i for i in range(len(self.imgs))}

    def _get_image_name(self, index):
        """
        Get name of the image indexed at `index`.
        :param index: the index of an image.
        :type: int
        :return: the name of the image file.
        :type: str
        """
        file_path = self.imgs[index][0]
        filename = os.path.split(file_path)[-1]
        return filename.split('.')[0]  # remove file extension

    def filter_image_indices(self, pattern):
        """
        Get a list of dataset indices for all images with names matching the given pattern.
        :param pattern: the pattern that returned images names must match.
        :type: str
        :return: a list of images that match the pattern.
        :type: List[Tuple]
        """
        return [ # create a list of indices
            i for i, path_class in enumerate(self.imgs) # return index
            if re.match(pattern, os.path.split(path_class[0])[-1])] # if last part of path matches regex

    @dispatch((int, torch.Tensor))  # single argument, either <int> or <Tensor>
    def __getitem__(self, index):
        """
        Dispatch to `super` method.
        """
        return super()[index]

    @dispatch(str)  # single argument, <str>
    def __getitem__(self, filename):
        """
        Get image index from filename, and dispatch to `super` method.
        """
        return super()[self._imgs_dict[filename]]


class FilePathIndexedMixin(ImageFolderMixin):
    """
    Mixin class for getting dataset items from a file path. It is intended to be used for retrieval tasks.
    To support indexing by index and path simultaneously, the `__getitem__` function was transformed to a single
    dispatch generic function using the `multiple-dispatch` module.
    Reference:
        1. [Python dispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch)
        2. [multiple-dispatch module](https://multiple-dispatch.readthedocs.io/en/latest/index.html)
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a image index dictionary with file paths as keys and dataset indices
        as values for efficient retrieval after initialization.
        :param args: super class arguments.
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self._imgs_dict = {tup[0]: i for i, tup in enumerate(self.imgs)}

    @dispatch((int, torch.Tensor))  # single argument, either <int> or <Tensor>
    def __getitem__(self, index):
        """
        Dispatch to `super` method.
        """
        return super()[index]

    @dispatch(str)  # single argument, <str>
    def __getitem__(self, file_path):
        """
        Get image index from file path, and dispatch to `super` method.
        """
        return super()[self._imgs_dict[file_path]]


class MultimodalEntityMixin(MultimodalDatasetMixin, ABC):
    """
    Dataset mixin for datasets in which the same entity is available in different modes (for example an image and its
    textual annotation, or a photo and a sketch of that same photo). The resulting dataset is defined with a base mode
    (the base dataset of the inherited `MultimodalDataset`). A tuple with one element from each mode is returned on
    `__getitem__` (one from the base dataset and one from each additional paired dataset). If more than one instance of
    the same entity is available for the same mode, a random instance is picked. Thus, the length of the `__getitem__`
    tuple is the same as the number of modes, and each tuple item has a `shape[0]` of `batch_size` (the remaining
    dimensions will depend on the format of each mode).
    NOTE: we assume that the same entity in a different mode will be contained in a file with a name that contains the
    name of the entity in the base dataset.
    """
    def __init__(self, base_dataset, *paired_datasets):
        """
        :param base_dataset: `MultimodalDataset` base dataset.
        :type: torch.utils.data.DatasetFolder
        :param paired_datasets: DatasetFolder object containing the entities of the base dataset, in different modes.
        :type: List[torchvision.datasets.DatasetFolder]
        """
        super().__init__(base_dataset)
        self.paired_datasets = paired_datasets
        self.entity_indexes = self._entity_indices()

    @property
    def cache_file_path(self):
        """
        File path of a `MultimodalEntityDataset` cache file.
        :return: the file path of the cache file.
        :type: str
        """
        return get_cache_directory(self.cache_filename)

    @property
    def cache_filename(self):
        """
        Filename of a `MultimodalEntityDataset` cache file.
        :return: the filename string.
        :type: str
        """
        return '{}.pickle'.format('-'.join([dataset.__class__.__name__ for dataset in [self, *self.paired_datasets]]))

    @property
    def _create_entity_indices(self):
        """
        Create the entity indices for the dataset: a list of dictionaries, one for each paired dataset, that contains
        the indices of all the elements in each corresponding paired dataset that correspond to the same entity in the
        base dataset, with base dataset element indices as keys.
        NOTE: this takes about 30 min. on a notebook i7. This could be optimized with multiprocessing, but it wasn't
        worth it at the time.
        :return: the entity indices object for the database.
        :type: List[Dict[int, List[int]]]
        """
        entity_indices = [] # contains a list for each base_dataset sample
        desc = 'Creating entity indices.'
        for i, base_sample in tqdm(enumerate(self.base_dataset.samples), desc=desc, total=len(self.base_dataset)):
            entity_indices.append([]) # contains a list for each paired_dataset
            pattern = self._get_filename(base_sample)
            for j, paired_dataset in enumerate(self.paired_datasets):
                entity_indices[i].append( # add list with all pattern matches in paired_datasets[j]
                    [k for k, sample in enumerate(paired_dataset.samples) if re.search(pattern, sample[0])])
        return entity_indices

    @staticmethod
    def _get_filename(element):
        """
        Get the filename of the given element.
        :param element: a `DatasetFolder` element tuple. Assumes the standard tuple format, with the file path in the
        first tuple index.
        :type: tuple
        :return: the file name of the element tuple
        :type: str
        """
        file_path = element[0]
        filename = os.path.split(file_path)[-1]
        return filename.split('.')[0] # remove file extension

    def _entity_indices(self):
        """
        Returns the entity indices object. The method tries to load the entity indices from cache, if available, and
        otherwise creates and caches it.
        :return: the entity indices object for the database.
        :type: List[Dict[int, List[int]]]
        """
        try:
            entity_indices = pickle.load(open(self.cache_file_path, 'rb'))
        except FileNotFoundError:
            entity_indices = self._create_entity_indices
            pickle.dump(entity_indices, open(self.cache_file_path, 'wb'))
        return entity_indices

    def __getitem__(self, index):
        """
        Override: return the item at `index` in the base dataset, along with a random instance of the same element in
        each of the modes defined by the different paired datasets.
        """
        return (self.base_dataset[index],
                *[dataset[choice(self.entity_indexes[index][i])] for i, dataset in enumerate(self.paired_datasets)])


class OneHotEncodingMixin:
    """
    Mixin class for adding unique one-hot encoding descriptors for each element in the dataset.
    """
    targets: List[int]

    def __init__(self, *args, **kwargs):
        """
        Initialize de base Dataset class and create a tensor with all one hot encodings.
        :param args: super class arguments.
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.encodings = torch.eye(len(self.targets))

    def _get_one_hot_encoding(self, index):
        """
        Get the one-hot-encoding of the given index.
        :param index: the index in decimal form.
        :type: int
        :return: a torch tensor with the one-hot-encoding form of the index.
        :type: torch.Tensor
        """
        return self.encodings[index]

    def __getitem__(self, item):
        return self._get_one_hot_encoding(item), self.targets[item]


class SiameseMixin(DatasetFolderMixin, ABC):
    """
    Mixin class for loading random pairs on `__getitem__` for any `DatasetFolder` dataset.
    """
    def __init__(self, *args, positive_pair_proportion=.5, **kwargs):
        """
        :param args: super class arguments.
        :type: list
        :param positive_pair_proportion: proportion of pairs that will be positive (same class).
        :type: float
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.target_probabilities = [positive_pair_proportion, 1 - positive_pair_proportion] # siamese target value prob

    def _get_pair(self, first_item_class_index):
        """
        Get a siamese pair from the paired dataset, which will be randomly positive (same class) or negative.
        :param first_item_class_index: the index of the first siamese pair element's class.
        :type: int
        :return: an item tuple
        :type: tuple
        """
        target = choice([0, 1], p=self.target_probabilities) # if `target==0` ...
        negative_classes = self._negative_classes(first_item_class_index)
        paired_item_cls = choice(negative_classes) if target else first_item_class_index # ... generate a positive pair
        return self._get_random_paired_item(paired_item_cls)

    @abstractmethod
    def _get_random_paired_item(self, class_index):
        """
        Get a random element belonging to the specified class index to be paired with the item requested on the
        `__getitem__` call.
        :param class_index: the index of the class to which the returned item must belong.
        :type: int
        :return: a random item belonging to `class_index`.
        :type: torch.Tensor
        """
        pass

    def _negative_classes(self, class_index):
        """
        Return a list with all dataset classes that are not `class_index`.
        :param class_index: the positive class index.
        :type: int
        :return: a list with all negative class indices.
        :type: List[int]
        """
        classes = list(self.class_to_idx.values())
        classes.remove(class_index)
        return classes

    def __getitem__(self, index):
        """
        Modify the Dataset's `__getitem__` method, returning each requested item along with another random item from
        the Dataset. The pair will be randomly positive (same class) or negative.
        :param index: an item's index.
        :type: int
        :return: a 2-tuple with the item corresponding to `index`, along with another random item, randomly positive or
        negative, according to `target_probabilities`.
        :type: tuple
        """
        item = super(SiameseMixin, self).__getitem__(index)
        item_class_index = self.targets[index]
        return item, self._get_pair(item_class_index)


class SiameseSingleDatasetMixin(ClassIndicesMixin, SiameseMixin):
    """
    SiameseMixin for use on a single Dataset instance.
    """
    @overrides
    def _get_random_paired_item(self, class_index):
        return super(SiameseMixin, self).__getitem__(choice(self.get_class_element_indices(class_index)))


class TripletMixin(DatasetFolderMixin):
    """
    Mixin class for loading online triplets on `__getitem__` for any `DatasetFolder` dataset.
    """
    def _get_random_negative(self, anchor_class_index):
        """
        Get a random negative triplet element of a random class for an anchor belonging to the specified class index.
        :param anchor_class_index: the class index of the triplet anchor.
        :type: int
        :return: a negative item tuple.
        :type: tuple
        """
        return self._get_random_triplet_item(choice(self._negative_classes(anchor_class_index)))

    def _get_random_positive(self, anchor_class_index):
        """
        Get a random positive triplet element for an anchor belonging to the specified class index.
        :param anchor_class_index: the class index of the triplet anchor.
        :type: int
        :return: an item tuple.
        :type: tuple
        """
        return self._get_random_triplet_item(anchor_class_index)

    @abstractmethod
    def _get_random_triplet_item(self, class_index):
        """
        Get a random element belonging to the specified class index to be included in a triplet along with the item
        requested on the `__getitem__` call. Thus, the returned item will be either the positive or negative triplet
        element.
        :param class_index: the class index.
        :type: int
        :return: a positive item tuple.
        :type: tuple
        """
        pass

    def _negative_classes(self, class_index):
        """
        Return a list with all dataset classes that are not `class_index`.
        :param class_index: the positive class index.
        :type: int
        :return: a list with all negative class indices.
        :type: List[int]
        """
        classes = list(self.class_to_idx.values())
        classes.remove(class_index)
        return classes

    def __getitem__(self, index):
        """
        Return a triplet consisting of an anchor (the indexed item), a positive (a random example of a different class),
        and a negative (a random example of the same class).
        :param index: an item's index.
        :type: int
        :return: a 3-tuple with the anchor, positive and negative.
        :type: tuple
        """
        anchor = super(TripletMixin, self).__getitem__(index)
        anchor_class_index = self.targets[index]
        return anchor, self._get_random_positive(anchor_class_index), self._get_random_negative(anchor_class_index)


class TripletSingleDatasetMixin(ClassIndicesMixin, TripletMixin):
    """
    TripletMixin for use on a single Dataset instance.
    """
    @overrides
    def _get_random_triplet_item(self, class_index):
        return super(TripletMixin, self).__getitem__(choice(self.get_class_element_indices(class_index)))
