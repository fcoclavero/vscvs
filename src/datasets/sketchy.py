import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from settings import DATA_SETS


class Sketchy(ImageFolder):
    """
    Utility class for loading the sketchy dataset. It's original structure is compatible with
    the torch ImageFolder, so I will just subclass that and apply some transforms.
    """
    def __init__(self, dataset):
        """
        Initialize the ImageFolder and perform transforms. Note that sketches and photos have the
        same exact dimension in both the sketchy and sketchy_test datasets.
        :param dataset: the root dir for photos or sketches.
        :type: str
        """
        super(Sketchy, self).__init__(
            root=dataset,
            transform=transforms.Compose([
                transforms.Resize(DATA_SETS['sketchy']['dimensions'][0]),
                transforms.CenterCrop(DATA_SETS['sketchy']['dimensions'][0]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )


class SketchyImageNames(ImageFolder):
    """
    Same as the Sketchy Dataset above, but each item is a tuple containing the image, it's class, and
    also it's name. This is used by the SketchyMixedBatches Dataset to find the sketches associated
    with each photo.
    """
    def __init__(self, dataset):
        """
        Initialize the ImageFolder and perform transforms. Note that sketches and photos have the
        same exact dimension in both the sketchy and sketchy_test datasets.
        :param dataset: the root dir for photos or sketches.
        :type: str
        """
        super(SketchyImageNames, self).__init__(
            root=dataset,
            transform=transforms.Compose([
                transforms.Resize(DATA_SETS['sketchy']['dimensions'][0]),
                transforms.CenterCrop(DATA_SETS['sketchy']['dimensions'][0]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

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

    def __getitem__(self, index):
        """
        Override of the Dataset method. Gets an image's pixel matrix, class, and name
        :param index: the index of an image
        :type: int
        :return: a tuple with the image's pixel matrix, class and name
        :type: tuple(torch.Tensor, int, str)
        """
        # tuple concatenation: https://stackoverflow.com/a/8538676
        return super(SketchyImageNames, self).__getitem__(index) + (self.__get_image_name__(index),)


class SketchyMixedBatches(Dataset):
    """
    Utility class for iterating over the sketchy dataset in our GAN approach. We will need both photos
    and sketches in each batch iteration, thus I will modify the Sketchy Dataset above to return an
    image and a sketch on each getitem so that each batch contains equal amounts of photos and sketches.
    The complete dataset has 100 photos per class, so a total of 125 * 100 = 12500 images. For each of
    these photos, there are a variable number of sketches (anywhere from 5 to 10) which have the same
    filename as the photo they are based on, plus '-n' where 'n' indicates the sketch number for that
    particular photo.
    """
    def __init__(self, dataset_name):
        """
        Initialize the ImageFolder and perform transforms. Note that sketches and photos have the
        same exact dimension in both the sketchy and sketchy_test datasets.
        :param dataset_name: the version of the sketchy dataset, either 'sketchy' or 'sketchy_test'
        :type: str
        """
        self.image_dataset = SketchyImageNames(DATA_SETS[dataset_name]['photos'])
        self.sketch_dataset = SketchyImageNames(DATA_SETS[dataset_name]['sketches'])

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        image, cls, name = self.image_dataset[index]
        return image, cls