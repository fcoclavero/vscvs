__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Datasets for managing multimodal data loading. """


import os
import pickle
import re

from numpy.random import choice
from overrides import overrides
from torch.utils.data import Dataset
from tqdm import tqdm

from vscvs.datasets.mixins import SiameseMixin, TripletMixin
from vscvs.utils import get_cache_directory


class MultimodalDataset(Dataset):
    """
    Dataset class for loading elements from multiple datasets on `__getitem__`. The first Dataset is used as base: the
    length of the multimodal Dataset corresponds to that of the base dataset, and the first item in the tuples returned
    on `__getitem__` corresponds to item with the requested index in the base dataset.
    """
    def __init__(self, base_dataset, *args, **kwargs):
        """
        :param base_dataset: base of the multimodal Dataset. It's length of the multimodal Dataset corresponds to this
        dataset's length, and the first item in the pairs returned on `__getitem__` corresponds to item with the
        requested index in this dataset.
        :type: torch.utils.data.Dataset
        :param args: additional arguments.
        :type: list
        :param kwargs: additional keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.base_dataset = base_dataset

    @overrides
    def __len__(self):
        return len(self.base_dataset)


class MultimodalDatasetFolder(MultimodalDataset):
    """
    MultimodalDataset subclass to be used with DatasetFolders.
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: base class arguments.
        :type: list
        :param kwargs: base class keyword arguments.
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx


class MultimodalEntityDataset(MultimodalDataset):
    """
    Dataset class for datasets in which the same entity is available in different modes (for example an image and its
    textual annotation, or a photo and a sketch of that same photo). The `MultimodalEntityDataset` is defined with a
    base mode (the base dataset of the inherited `MultimodalDataset`). A tuple with one element from each mode is
    returned on `__getitem__` (one from the base dataset and one from each additional paired dataset). If more than one
    instance of the same entity is available for the same mode, a random instance is picked. Thus, the length of the
    `__getitem__` tuple is the same as the number of modes, and each tuple item has a `shape[0]` of `batch_size` (the
    remaining dimensions will depend on the format of each mode).
    NOTE: we assume that the same entity in a different mode will be contained in a file with a name that contains the
    name of the entity in the base dataset.
    """
    def __init__(self, base_dataset, *paired_datasets):
        """
        :param base_dataset: `MultimodalDataset` base dataset.
        :type: torch.utils.data.DatasetFolder
        :param paired_datasets: DatasetFolder object containing the entities of the base dataset, in different modes.
        :type: list<torchvision.datasets.DatasetFolder>
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
        :type: list<dict<int: list<int>>>
        """
        entity_indices = [] # contains a list for each base_dataset sample
        for i, base_sample in tqdm(enumerate(self.base_dataset.samples), desc='Creating indices.', total=len(self)):
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
        :type: list<dict<int: list<int>>>
        """
        try:
            entity_indices = pickle.load(open(self.cache_file_path, 'rb'))
        except FileNotFoundError:
            entity_indices = self._create_entity_indices
            pickle.dump(entity_indices, open(self.cache_file_path, 'wb'))
        return entity_indices

    @overrides
    def __getitem__(self, index):
        """
        Override: return the item at `index` in the base dataset, along with a random instance of the same element in
        each of the modes defined by the different paired datasets.
        """
        return (self.base_dataset[index],
                *[dataset[choice(self.entity_indexes[index][i])] for i, dataset in enumerate(self.paired_datasets)])


class SiameseDataset(SiameseMixin, MultimodalDatasetFolder):
    """
    Dataset class for loading random online pairs on `__getitem__` for the given pair of Datasets. The first Dataset is
    used as base: the length of the siamese Dataset corresponds to that of the first dataset, and the first item in the
    pairs returned on `__getitem__` corresponds to item with the requested index in the base dataset (plus a random
    item from the second dataset).
    """
    def __init__(self, base_dataset, paired_dataset, *args, **kwargs):
        """
        :param base_dataset: base dataset for the `MultimodalDatasetFolder` constructor arguments.
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__getitem__`.
        It must have the same classes as `base_dataset` and must inherit `ClassIndicesMixin` for easy access to specific
        class elements.
        :type: torch.utils.data.Dataset + vscvs.datasets.mixins.ClassIndicesMixin
        :param args: super class arguments
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(base_dataset, *args, **kwargs)
        self.paired_dataset = paired_dataset

    @overrides
    def _get_random_paired_item(self, class_index):
        """
        Override: the random paired item belongs to the paired dataset.
        """
        item_index = choice(self.paired_dataset.get_class_element_indices(class_index))
        return self.paired_dataset[item_index]

    @overrides
    def __getitem__(self, index):
        """
        Override: the first pair item belongs to base dataset and the random paired item belongs to the paired dataset.
        """
        item = self.base_dataset[index]
        item_class = item[1]
        return item, self._get_pair(item_class)


class TripletDataset(TripletMixin, MultimodalDatasetFolder):
    """
    Dataset class for loading random online triplets on `__getitem__` for the given pair of Datasets. The first Dataset
    is used as base: the length of the triplet Dataset corresponds to that of the first dataset, and the first item in
    the triplets returned on `__getitem__` corresponds to the item with the requested index in the base dataset (the
    triplet anchor). Other two random items from the second dataset are drawn, one with the same class as the anchor
    (the positive) and one of a different random class (the negative).
    """
    def __init__(self, base_dataset, paired_dataset, *args, **kwargs):
        """
        :param base_dataset: base dataset for the `MultimodalDatasetFolder` constructor arguments.
        :type: torch.utils.data.Dataset
        :param paired_dataset: the Dataset from which random accompanying items will be drawn upon each `__getitem__`.
        It must have the same classes as `base_dataset` and must inherit `ClassIndicesMixin` for easy access to specific
        class elements.
        :type: torch.utils.data.Dataset + vscvs.datasets.mixins.ClassIndicesMixin
        :param args: super class arguments
        :type: list
        :param kwargs: super class keyword arguments.
        :type: dict
        """
        super().__init__(base_dataset, *args, **kwargs)
        self.paired_dataset = paired_dataset

    @overrides
    def _get_random_triplet_item(self, class_index):
        """
        Override: the random triplet item belongs to the paired dataset.
        """
        item_index = choice(self.paired_dataset.get_class_element_indices(class_index))
        return self.paired_dataset[item_index]

    @overrides
    def __getitem__(self, index):
        """
        Override: the anchor belongs to base dataset and the random positive and negative belong to the paired dataset.
        """
        anchor = self.base_dataset[index]
        anchor_class_index = anchor[1]
        return anchor, self._get_random_positive(anchor_class_index), self._get_random_negative(anchor_class_index)
