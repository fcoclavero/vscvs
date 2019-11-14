__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" DataSets for loading the Sketchy dataset with different options. """


import os
import re
import pickle
import torch

from tqdm import tqdm
from multipledispatch import dispatch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from settings import DATA_SOURCES
from src.datasets.mixins import TripletMixin, FilenameIndexedMixin, BinaryEncodingMixin


class Sketchy(ImageFolder):
    """
    Utility class for loading the sketchy dataset. It's original structure is compatible with
    the torch ImageFolder, so I will just subclass that and apply some transforms.
    """
    def __init__(self, root_directory, *custom_transforms, in_channels=3, **kwargs):
        """
        Initialize the ImageFolder and perform transforms. Note that sketches and photos have the
        same exact dimension in both the sketchy and sketchy_test datasets.
        :param root_directory: the root dir for photos or sketches.
        :type: str
        :param custom_transforms: additional transforms for the Dataset
        :type: torchvision.transforms
        :param in_channels: number of image color channels.
        :type: int
        """
        super().__init__(
            root=root_directory,
            transform=transforms.Compose([
                transforms.Resize(DATA_SOURCES['sketchy']['dimensions'][0]),
                transforms.CenterCrop(DATA_SOURCES['sketchy']['dimensions'][0]),
                *custom_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    list((0.5 for _ in range(in_channels))), # mean sequence for each channel
                    list((0.5 for _ in range(in_channels))) # std sequence for each channel
                ),
            ])
        )


class SketchyTriplets(TripletMixin, Sketchy):
    """
    Sketchy Dataset with online triplet generation.
    """
    pass


class SketchyFilenameIndexed(FilenameIndexedMixin, Sketchy):
    """
    Sketchy Dataset with additional filename indexation.
    """
    pass


class SketchyBinaryEncoded(BinaryEncodingMixin, Sketchy):
    """
    Sketchy Dataset with additional filename indexation.
    """
    pass


class SketchyImageNames(Sketchy):
    """
    Same as the Sketchy Dataset above, but each item is a tuple containing the image, it's class, and
    also it's name. This is used by the SketchyMixedBatches Dataset to find the sketches associated
    with each photo.
    """
    def __get_image_name__(self, index):
        """
        Get name of the image indexed at `index`. This id can the be used to find the sketches
        associated with the image.
        :param index: the index of an image
        :type: int
        :return: the name of the image: it's id
        :type: str
        """
        path = self.imgs[index][0]
        filename = os.path.split(path)[-1]
        return filename.split('.')[0] # remove file extension

    @dispatch((int, torch.Tensor)) # single argument, either <int> or <Tensor>
    def __getitem__(self, index):
        """
        Get an image's pixel matrix, class, and name from its positional index.
        To support indexing by position and name simultaneously, this function was
        transformed to a single dispatch generic function using the multiple-dispatch module.
        https://docs.python.org/3/library/functools.html#functools.singledispatch
        https://multiple-dispatch.readthedocs.io/en/latest/index.html
        :param index: the index of an image
        :type: int
        :return: a tuple with the image's pixel matrix, class and name
        :type: tuple(torch.Tensor, int, str)
        """
        # tuple concatenation: https://stackoverflow.com/a/8538676
        return super().__getitem__(index) + (self.__get_image_name__(index),)

    @dispatch(str)  # single argument, <str>
    def __getitem__(self, name):
        """
        Get an image's pixel matrix, class, and name from its name.
        To support indexing by position and name simultaneously, this function was
        transformed to a single dispatch generic function using the multiple-dispatch module.
        https://docs.python.org/3/library/functools.html#functools.singledispatch
        https://multiple-dispatch.readthedocs.io/en/latest/index.html
        :param name: the image name
        :type: str
        :return: a tuple with the image's pixel matrix, class and name
        :type: tuple(torch.Tensor, int, str)
        """
        index = next( # stop iterator on first match and return index
            i for i, path_class in enumerate(self.imgs) # return index
            if re.match(name, os.path.split(path_class[0])[-1]) # if last part of path matches regex
        )
        return super().__getitem__(index) + (name,) # tuple concatenation

    def get_image_indexes(self, pattern):
        """
        Get a list of dataset indexes for all images matching the given pattern.
        :param pattern: the pattern that returned images names must match
        :type: str
        :return: a list of images' pixel matrix
        :type: list<torch.Tensor>
        """
        return [ # create a list of indices
            i for i, path_class in enumerate(self.imgs) # return index
            if re.match(pattern, os.path.split(path_class[0])[-1]) # if last part of path matches regex
        ]

    def get_images(self, pattern):
        """
        Get a list of pixel matrices for all images matching the given pattern.
        :param pattern: the pattern that returned images names must match
        :type: str
        :return: a list of images' pixel matrix
        :type: list<torch.Tensor>
        """
        return [self.__getitem__(index)[0] for index in self.get_image_indexes(pattern)]


class SketchyMixedBatches(Dataset):
    """
    Utility class for iterating over the sketchy dataset in our GAN approach. We will need both photos
    and sketches in each batch iteration, thus I will modify the Sketchy Dataset above to return an
    image and a sketch on each getitem so that each batch contains equal amounts of photos and sketches.
    The complete dataset has 100 photos per class, so a total of 125 * 100 = 12500 images. For each of
    these photos, there are a variable number of sketches (anywhere from 5 to 10) which have the same
    filename as the photo they are based on, plus '-n' where 'n' indicates the sketch number for that
    particular photo.
    We train with batches containing the sketches based on the batches' photos because this should be
    the hardest case for the discriminator.
    """
    def __init__(self, dataset_name, *custom_transforms, **kwargs):
        """
        Initialize the ImageFolder and perform transforms. Note that sketches and photos have the
        same exact dimension in both the sketchy and sketchy_test datasets.
        :param dataset_name: the version of the sketchy dataset, either 'sketchy' or 'sketchy_test'
        :type: str
        :param custom_transforms: additional transforms for the Dataset
        :type: torchvision.transforms
        """
        self.photos_dataset = SketchyImageNames(DATA_SOURCES[dataset_name]['photos'], *custom_transforms)
        self.sketch_dataset = SketchyImageNames(DATA_SOURCES[dataset_name]['sketches'], *custom_transforms)
        try:
            # creating the reference list for the complete dataset is really expensive, so we
            # try to load from pickle. If pickle not available, the list is created and then pickled
            self.__sketches__ = pickle.load(open(r'data\image_sketch_indexes.pickle', 'rb'))
        except Exception as e:
            self.__sketches__ = [  # list that contains a list of sketches for each photo in the dataset
                self.sketch_dataset.get_image_indexes(photo[2]) for photo in tqdm(self.photos_dataset)
            ]
            pickle.dump(self.__sketches__, open(r'data\image_sketch_indexes.pickle', 'wb'))

    def __len__(self):
        """
        Get the length fo the Dataset, which in this case is defined as the amount of photos.
        :return: the length of the Dataset
        :type: int
        """
        return len(self.photos_dataset)

    def __getitem__(self, index):
        """
        Return a photo, all sketches based on it, and its class given the photo's index.
        :param index: the photo's index
        :type: int
        :return: a tuple with the photos's pixel matrix, the associated sketches' pixel
        matrices, and the images' class
        :type: tuple(torch.Tensor, list<torch.Tensor>, int)
        """
        photo, cls, name = self.photos_dataset[index]
        return photo, torch.stack([self.sketch_dataset[i][0] for i in self.__sketches__[index]]), cls
