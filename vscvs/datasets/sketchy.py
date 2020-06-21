__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" DataSets for loading the Sketchy dataset with different options. """


import os
import pickle

from torchvision import transforms
from torchvision.datasets import ImageFolder

from settings import DATA_SOURCES

from .mixins import BinaryEncodingMixin
from .mixins import ClassIndicesMixin
from .mixins import FileNameIndexedMixin
from .mixins import FilePathIndexedMixin
from .mixins import OneHotEncodingMixin
from .mixins import SiameseSingleDatasetMixin
from .mixins import TripletSingleDatasetMixin


class Sketchy(ImageFolder):
    """
    Utility class for loading the sketchy dataset. It's original structure is compatible with
    the torch ImageFolder, so I will just subclass that and apply some transforms.
    """

    def __init__(self, image_data_source, *custom_transforms, in_channels=3, normalize=True, size=None, **__):
        """
        NOTE: sketches and photos have the same exact dimension in both the `sketchy` and `sketchy_test` datasets.
        :param image_data_source: the DATA_SOURCE name for images
        :type: str
        :param custom_transforms: additional transforms for the Dataset
        :type: torchvision.transforms
        :param in_channels: number of image color channels.
        :type: int
        :param normalize: whether to normalize the image tensor. This is recommended for training any model.
        :type: bool
        :param size: desired output size for dataset images. If `size` is a sequence like `(h, w)`, the output size will
        be matched to this. If size is an int, the smaller edge of the image will be matched to this number. i.e, if
        `height > width`, then image will be rescaled to `(size * height / width, size)`.
        :type: Union[Tuple[int, int], int]
        """
        transforms_list = [
            transforms.Resize(size or DATA_SOURCES[image_data_source]["dimensions"]),
            transforms.CenterCrop(size or DATA_SOURCES[image_data_source]["dimensions"]),
            *custom_transforms,
            transforms.ToTensor(),
        ]
        if normalize:
            transforms_list.append(
                transforms.Normalize(
                    list((0.5 for _ in range(in_channels))),  # mean sequence for each channel
                    list((0.5 for _ in range(in_channels))),
                )
            )  # std sequence for each channel
        super().__init__(root=DATA_SOURCES[image_data_source]["images"], transform=transforms.Compose(transforms_list))

    @property
    def classes_dataframe(self):
        return pickle.load(open(os.path.join(self.root, "classes.pickle"), "rb"))


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
