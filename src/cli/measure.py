__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image retrieval benchmarks. """


import click


@click.group()
def measure():
    """ Image retrieval benchmarks click group. """
    pass


@measure.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice([
        d + '_filenames' for d in ['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches']
    ])
)
@click.option('--embeddings_name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--test_split', prompt='Test split', default=.8, help='Proportion of the dataset to be used for queries.')
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def class_recall(dataset_name, embeddings_name, test_split, k, n_gpu):
    click.echo('Calculating recall@{} for {} embeddings'.format(k, embeddings_name))
    from src.utils.embeddings import average_class_recall
    average_class_recall(dataset_name, embeddings_name, k, n_gpu)
