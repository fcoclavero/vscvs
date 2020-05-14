__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Create tensorboard visualizations. """


import click
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from vscvs.utils import get_image_directory


@click.group()
def tensorboard():
    """ Display in Tensorboard. """
    pass


@tensorboard.command()
@click.option('--path', prompt='Full path.', help='The full path to the image to be visualized.')
def image(path):
    """ Add the image in the given path to Tensorboard. """
    from PIL import Image
    image_name = os.path.basename(path)
    transform = transforms.ToTensor()
    writer = SummaryWriter(get_image_directory('show image'))
    writer.add_image(image_name, transform(Image.open(path)), 0)
    writer.close()
    click.echo('Image added to Tensorboard: {}'.format(image_name))
