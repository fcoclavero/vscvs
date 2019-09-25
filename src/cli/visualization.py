__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image visualization. """


import click


@click.group()
def visualization():
    """ Image visualization click group. """
    pass


@visualization.command()
@click.option(
    '--path', prompt='Full path.', help='The full path to the image to be visualized.'
)
def show_image(path):
    """ Display the image in the given path. """
    from PIL import Image
    image = Image.open(path)
    image.show()


@visualization.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--batch_size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
def show_sample_batch(dataset_name, batch_size, workers):
    from src.visualization import display_sample_batch
    display_sample_batch(dataset_name, batch_size, workers)
