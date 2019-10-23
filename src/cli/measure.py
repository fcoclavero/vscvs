__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'
 
 
""" Image retrieval benchmarks. """
 
 
import click
 
from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
 
 
@click.group()
def measure():
    """ Image retrieval benchmarks click group. """
    pass
 
 
@measure.group()
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def recall(context, **kwargs):
    """ Image recall benchmarks click group. """
    pass
 
 
@recall.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset that corresponds to the given embeddings.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--embeddings-name', prompt='Embeddings name', help='Name of the embeddings directory.')
@click.option('--test-split', prompt='Test split', default=.2, help='Proportion of the dataset to be used for queries.')
def same_class(_, dataset_name, embeddings_name, test_split, k, n_gpu):
    click.echo('Calculating class recall@{} for {} embeddings'.format(k, embeddings_name))
    from src.utils.embeddings import average_class_recall
    average_class_recall(dataset_name, embeddings_name, test_split, k, n_gpu)
 
 
@measure.group()
def cross_modal():
    """ Cross modal retrieval benchmarks click group. """
    pass
 
 
@cross_modal.group()
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def recall(context, **kwargs):
    """ Image recall benchmarks click group. """
    pass
 
 
# @cross_modal.command()
# @pass_context_to_kwargs
# @click.option(
#     '--sketch-dataset-name', type=click.Choice(['sketchy-sketches', 'sketchy-test-sketches']),
#     prompt='Sketch dataset name', help='The name of the dataset that corresponds to the sketch embeddings.',
# )
# @click.option(
#     '--photo-dataset-name', type=click.Choice(['sketchy-photos', 'sketchy-test-photos']),
#     prompt='Photos dataset name', help='The name of the dataset that corresponds to the photo embeddings.',
# )
# @click.option('--sketch-embeddings-name', prompt='Sketch embeddings name', help='Name of sketch embeddings directory.')
# @click.option('--photo-embeddings-name', prompt='Photo embeddings name', help='Name of photo embeddings directory.')
# def same_class(_, sketch_dataset_name, photo_dataset_name, sketch_embeddings_name, photo_embeddings_name,
#                test_split, k, n_gpu):
#     click.echo('Calculating cross modal class recall@{} for the {} and {} embeddings'.format(
#         k, sketch_embeddings_name,photo_embeddings_name
#     ))
#     from src.utils.embeddings import cross_modal_average_class_recall
#     cross_modal_average_class_recall(
#         sketch_dataset_name, photo_dataset_name, sketch_embeddings_name, photo_embeddings_name, test_split, k, n_gpu
#     )