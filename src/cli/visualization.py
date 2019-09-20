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