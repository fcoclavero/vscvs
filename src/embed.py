__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click

from src.utils.embeddings import create_embeddings


@click.group()
def embed():
    """ Image embedding creation click group. """
    pass


@embed.command()
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings will be saved.'
)
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--batch_size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(embedding_directory_name, dataset_name, batch_size, workers, n_gpu):
    click.echo('HOG embeddings for {} dataset'.format(dataset_name))
    from src.models.hog import HOG
    create_embeddings(embedding_directory_name, dataset_name, HOG(), batch_size, workers, n_gpu)
