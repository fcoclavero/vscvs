__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" DataSets for loading the Sketchy dataset with different options. """


import os
import pickle

from torchvision import transforms
from torchvision.datasets import ImageFolder

from .mixins import BinaryEncodingMixin, ClassIndicesMixin, FileNameIndexedMixin, FilePathIndexedMixin, \
    OneHotEncodingMixin,  SiameseSingleDatasetMixin, TripletSingleDatasetMixin
from settings import DATA_SOURCES


class Sketchy(ImageFolder):
    """
    Utility class for loading the sketchy dataset. It's original structure is compatible with
    the torch ImageFolder, so I will just subclass that and apply some transforms.
    """
    def __init__(self, image_data_source, *custom_transforms, in_channels=3, **__):
        """
        NOTE: sketches and photos have the same exact dimension in both the `sketchy` and `sketchy_test` datasets.
        :param image_data_source: the DATA_SOURCE name for images
        :type: str
        :param custom_transforms: additional transforms for the Dataset
        :type: torchvision.transforms
        :param in_channels: number of image color channels.
        :type: int
        """
        super().__init__(
            root=DATA_SOURCES[image_data_source]['images'],
            transform=transforms.Compose([
                transforms.Resize(DATA_SOURCES[image_data_source]['dimensions'][0]),
                transforms.CenterCrop(DATA_SOURCES[image_data_source]['dimensions'][0]),
                *custom_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    list((0.5 for _ in range(in_channels))), # mean sequence for each channel
                    list((0.5 for _ in range(in_channels))))])) # std sequence for each channel

    @property
    def classes_dataframe(self):
        return pickle.load(open(os.path.join(self.root, 'classes.pickle'), 'rb'))


class SketchyClassIndices(ClassIndicesMixin, Sketchy):
    """
    Sketchy Dataset with class indices.
    """
    pass


class SketchySiamese(SiameseSingleDatasetMixin, Sketchy):
    """
    Sketchy Dataset with online siamese pair generation.
    """
    pass


class SketchyTriplets(TripletSingleDatasetMixin, Sketchy):
    """
    Sketchy Dataset with online triplet generation.
    """
    pass


class SketchyNamePathIndexed(FileNameIndexedMixin, Sketchy):
    """
    Sketchy Dataset with additional filename indexation.
    """
    pass


class SketchyFilePathIndexed(FilePathIndexedMixin, Sketchy):
    """
    Sketchy Dataset with additional file path indexation.
    """
    pass


class SketchyBinaryEncoded(BinaryEncodingMixin, Sketchy):
    """
    Sketchy Dataset with additional binary encodings for each item.
    """
    pass


class SketchyOneHotEncoded(OneHotEncodingMixin, Sketchy):
    """
    Sketchy Dataset with additional one hot encodings for each item.
    """
    pass
