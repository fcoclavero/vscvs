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
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings will be saved.'
)
@click.option('--in_channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell_size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--n_bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed_gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
@click.option('--batch_size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(dataset_name, embedding_directory_name, in_channels, cell_size, n_bins, signed_gradients,
        batch_size, workers, n_gpu):
    click.echo('HOG embeddings for {} dataset'.format(dataset_name))
    from src.models.hog import HOG
    model = HOG(in_channels, cell_size, n_bins, signed_gradients)
    create_embeddings(model, dataset_name, embedding_directory_name, batch_size, workers, n_gpu)
