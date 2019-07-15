__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click

from .utils import get_device


@click.group()
def embed():
    """ Image embedding creation click group. """
    pass


@embed.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(dataset_name, n_gpu):
    click.echo('HOG embeddings for %s dataset' % dataset_name)

    from src.models.hog import HOG

    device = get_device(n_gpu)
    model = HOG