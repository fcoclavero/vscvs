__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Miscellaneous resource creation. """


import click


@click.group()
def create():
    """ Miscellaneous resource creation click group. """
    pass


@create.command()
@click.option('--dataset-name', prompt='Dataset name', help='Name of the dataset for which classes must be created.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches']))
@click.option('--distance', prompt='Distance', type=click.Choice(['cosine', 'pairwise']),
              help='The distance measure to be used for pre-computing class document vector distances.')
@click.option('--tsne-dimension', default=2, help='The target dimensionality for the lower dimension projection.')
def classes(dataset_name, distance, tsne_dimension):
    click.echo('Creating a new classes dataframe for the {} dataset'.format(dataset_name))
    from vscvs.preprocessing import create_classes_data_frame # import here to avoid loading word vectors on every command
    create_classes_data_frame(dataset_name, distance, tsne_dimension)


@create.command()
@click.option('--n', prompt='Number of samples', help='The number of sample vectors to be created.', type=int)
@click.option('--dimension', prompt='Sample dimensionality', help='The dimension of sample vectors.', type=int)
def sample_vectors(n, dimension):
    click.echo('Creating sample vectors.')
    from vscvs.preprocessing import create_sample_vectors
    create_sample_vectors(n, dimension)
