__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image visualization. """


import click


@click.group()
def show():
    """ Image visualization click group. """
    pass


@show.command()
@click.option(
    '--path', prompt='Full path.', help='The full path to the image to be visualized.'
)
def image(path):
    """ Display the image in the given path. """
    from PIL import Image
    image = Image.open(path)
    image.show()


@show.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--batch_size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
def sample_batch(dataset_name, batch_size, workers):
    from src.visualization import display_sample_batch
    display_sample_batch(dataset_name, batch_size, workers)


@show.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings will be saved.'
)
@click.option('--load_projection', prompt='Load projection', help='Try to load pickled TSNE projections', is_flag=True)
def embedding_tsne(dataset_name, embedding_directory_name, load_projection):
    click.echo('Display projection of the {} embeddings'.format(embedding_directory_name))
    from src.visualization import plot_embedding_tsne
    plot_embedding_tsne(dataset_name, embedding_directory_name, load_projection)
