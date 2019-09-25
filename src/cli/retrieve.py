__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image retrieval given a query image. """


import click

from src.utils.embeddings import query_embeddings


@click.group()
def retrieve():
    """ Image retrieval click group. """
    pass


@retrieve.command()
@click.option(
    '--query_image_filename', prompt='Query image full path.', help='The dataset will be queried for similar images.'
)
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice([
        d + '_filenames' for d in ['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches']
    ])
)
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings will be saved.'
)
@click.option('--in_channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell_size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--n_bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed_gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(query_image_filename, dataset_name, embedding_directory_name, in_channels, cell_size, n_bins, signed_gradients,
        k, n_gpu):
    click.echo('Querying {} embeddings'.format(embedding_directory_name))
    from src.models.hog import HOG
    model = HOG(in_channels, cell_size, n_bins, signed_gradients)
    query_embeddings(model, query_image_filename, dataset_name, embedding_directory_name, k, n_gpu)
