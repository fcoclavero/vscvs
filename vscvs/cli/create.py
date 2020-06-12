__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Miscellaneous resource creation. """


import click
import os

from tqdm import tqdm


@click.group()
def create():
    """ Miscellaneous resource creation. """
    pass


@create.command()
@click.option('--dataset-name', prompt='Dataset name', help='Name of the dataset for which classes must be created.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches',
                                 'sketchy-test-photos-multimodal', 'sketchy-test-sketches-multimodal']))
@click.option('--distance', prompt='Distance', type=click.Choice(['cosine', 'pairwise']),
              help='The distance measure to be used for pre-computing class document vector distances.')
@click.option('--tsne-dimension', default=2, help='The target dimensionality for the lower dimension projection.')
def classes(dataset_name, distance, tsne_dimension):
    """ Create a class name word vector dataframe for a dataset. """
    click.echo('Creating a new classes dataframe for the {} dataset'.format(dataset_name))
    from vscvs.preprocessing import create_classes_data_frame
    create_classes_data_frame(dataset_name, distance, tsne_dimension)


@create.command()
@click.option('--dataset-name', prompt='Dataset name', help='Name of the dataset for metadata creation.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches']))
def metadata_tsv(dataset_name):
    """ Create a class name word vector dataframe for a dataset. """
    click.echo('Creating {}.tsv for the Tensorboard embeddings projector.'.format(dataset_name))
    from vscvs.embeddings import create_metadata_tsv
    create_metadata_tsv(dataset_name)


@create.command()
@click.option('--n', prompt='Number of samples', help='The number of sample vectors to be created.', type=int)
@click.option('--dimension', prompt='Sample dimensionality', help='The dimension of sample vectors.', type=int)
def sample_vectors(n, dimension):
    """ Create a sample vectors dataset. """
    click.echo('Creating sample vectors.')
    from vscvs.preprocessing import create_sample_vectors
    create_sample_vectors(n, dimension)


@create.command()
@click.option('--photos-per-class', default=10, help='Number of photo image instances per class to be included.')
@click.option('--photos-root', prompt='Photos root', type=click.Path(exists=True),
              help='Directory containing the sketchy photo image files.')
@click.option('--sketches-root', prompt='Sketches root', type=click.Path(exists=True),
              help='Directory containing the sketchy sketch image files.')
@click.option('--test-photos-root', prompt='Test photos root', type=click.Path(exists=True),
              help='Directory containing the sketchy test photo image files.')
@click.option('--test-sketches-root', prompt='Test sketches root', type=click.Path(exists=True),
              help='Directory containing the sketchy test sketch1 image files.')
def sketchy_test_dataset(photos_per_class, photos_root, sketches_root, test_photos_root, test_sketches_root):
    """
    Create the sketchy test dataset, assuring each included instance is available as both photo and sketch (for
    multimodal training).
    """
    import re
    from shutil import copyfile

    for class_name in tqdm(os.listdir(photos_root), desc='Populating photos'):
        for test_photo_path in os.listdir(os.path.join(photos_root, class_name))[:photos_per_class]:
            if not os.path.exists(os.path.join(test_photos_root, class_name)):
                os.makedirs(os.path.join(test_photos_root, class_name))
            src = os.path.join(photos_root, class_name, test_photo_path)
            dst = os.path.join(test_photos_root, class_name, test_photo_path)
            copyfile(src, dst)

    for class_name in tqdm(os.listdir(sketches_root), desc='Populating sketches'):
        sketch_paths = os.listdir(os.path.join(sketches_root, class_name))
        for test_photo_path in os.listdir(os.path.join(test_photos_root, class_name)):
            if not os.path.exists(os.path.join(test_sketches_root, class_name)):
                os.makedirs(os.path.join(test_sketches_root, class_name))
            same_instance_sketch = next(path for path in sketch_paths if re.search(test_photo_path.split('.')[0], path))
            src = os.path.join(sketches_root, class_name, same_instance_sketch)
            dst = os.path.join(test_sketches_root, class_name, test_photo_path)
            copyfile(src, dst)
