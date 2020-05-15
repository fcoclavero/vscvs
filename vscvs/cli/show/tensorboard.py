__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Create tensorboard visualizations. """


import click
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from vscvs.embeddings import load_embedding_pickles
from vscvs.utils import get_embedding_directory, get_image_directory


@click.group()
def tensorboard():
    """ Display in Tensorboard. """
    pass


@tensorboard.command()
@click.option('--embeddings-name', prompt='Embedding directory', help='Static directory where embeddings are saved.')
@click.option('--tags', help='Optional tags for organizing embeddings.', multiple=True)
def embeddings(embeddings_name, tags):
    """ Add the embeddings in the given path to Tensorboard. """
    embeddings_tensor = load_embedding_pickles(embeddings_name)
    writer = SummaryWriter(get_embedding_directory(embeddings_name, tags))
    writer.add_embedding(embeddings_tensor) # TODO: add metadata
    writer.close()
    click.echo('Image added to Tensorboard: {}'.format(embeddings_name))


@tensorboard.command()
@click.option('--path', prompt='Full path.', help='The full path to the image to be visualized.')
@click.option('--tags', help='Optional tags for organizing embeddings.', multiple=True)
def image(path, tags):
    """ Add the image in the given path to Tensorboard. """
    from PIL import Image
    image_name = os.path.basename(path)
    transform = transforms.ToTensor()
    writer = SummaryWriter(get_image_directory('show image', tags))
    writer.add_image(image_name, transform(Image.open(path)), 0)
    writer.close()
    click.echo('Image added to Tensorboard: {}'.format(image_name))
