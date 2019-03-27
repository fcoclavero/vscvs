from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from settings import DATA_SETS


class SketchyImages(ImageFolder):
    def __init__(self):
        super(SketchyImages, self).__init__(
            root=DATA_SETS['sketchy']['images'],
            transform=transforms.Compose([
                transforms.Resize(DATA_SETS['sketchy_test']['dimensions'][0]),
                transforms.CenterCrop(DATA_SETS['sketchy_test']['dimensions'][0]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )