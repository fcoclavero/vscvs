__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Datasets for managing multimodal data loading. """


from numpy.random import choice
from overrides import overrides
from torch.utils.data import Dataset

from .mixins import SiameseMixin, SiameseSingleDatasetMixin, TripletMixin, MultimodalEntityMixin


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
        :type: List
        :param kwargs: additional keyword arguments.
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)


class MultimodalDatasetFolder(MultimodalDataset):
    """
    MultimodalDataset subclass to be used with DatasetFolders.
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: base class arguments.
        :type: List
        :param kwargs: base class keyword arguments.
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx
        self.samples = self.base_dataset.samples
        self.targets = self.base_dataset.targets


class MultimodalEntityDataset(MultimodalEntityMixin, MultimodalDataset):
    """
    MultimodalEntity dataset.
    """
    pass


class MultimodalEntityDatasetFolder(MultimodalEntityMixin, MultimodalDatasetFolder):
    """
    MultimodalEntityDataset for DatasetFolder datasets.
    """
    pass


class MultimodalEntitySiameseDataset(SiameseSingleDatasetMixin, MultimodalEntityDatasetFolder):
    """
    MultimodalEntityDataset subclass to load multimodal siamese pairs.
    """
    pass


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
        :type: List
        :param kwargs: super class keyword arguments.
        :type: Dict
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
        :type: List
        :param kwargs: super class keyword arguments.
        :type: Dict
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
