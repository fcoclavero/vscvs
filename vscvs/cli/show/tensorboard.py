__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Create tensorboard visualizations. """


import click
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from vscvs.embeddings import load_embeddings
from vscvs.utils import get_path


@click.group()
def tensorboard():
    """ Display in Tensorboard. """
    pass


@tensorboard.command()
@click.option('--embeddings-name', prompt='Embedding directory', help='Static directory where embeddings are saved.')
@click.option('-t', '--tag', help='Optional tags for organizing embeddings.', multiple=True)
def embeddings(embeddings_name, tags):
    """ Add the embeddings in the given path to Tensorboard. """
    embeddings_tensor = load_embeddings(embeddings_name)
    writer = SummaryWriter(get_path('tensorboard', 'embeddings', *tags, embeddings_name))
    writer.add_embedding(embeddings_tensor) # TODO: add metadata
    writer.close()
    click.echo('Embeddings added to Tensorboard: {}'.format(embeddings_name))


@tensorboard.command()
@click.option('--path', prompt='Full path.', help='The full path to the image to be visualized.')
@click.option('-t', '--tag', help='Optional tags for organizing embeddings.', multiple=True)
def image(path, tags):
    """ Add the image in the given path to Tensorboard. """
    from PIL import Image
    image_name = os.path.basename(path)
    transform = transforms.ToTensor()
    writer = SummaryWriter(get_path('tensorboard', 'images', tags))
    writer.add_image(image_name, transform(Image.open(path)), 0)
    writer.close()
    click.echo('Image added to Tensorboard: {}'.format(image_name))
