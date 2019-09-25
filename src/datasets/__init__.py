__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Register datasets here to make them available in the CLI. """

from settings import DATA_SOURCES

from src.datasets.sketchy import Sketchy, SketchyImageNames, SketchyMixedBatches, SketchyTriplets, \
                                 SketchyFilenameIndexed


DATASET_DATA_SOURCES = {
    'sketchy_photos': DATA_SOURCES['sketchy']['photos'],
    'sketchy_photos_triplets': DATA_SOURCES['sketchy']['photos'],
    'sketchy_photos_filenames': DATA_SOURCES['sketchy']['photos'],
    'sketchy_sketches': DATA_SOURCES['sketchy']['sketches'],
    'sketchy_sketches_triplets': DATA_SOURCES['sketchy']['sketches'],
    'sketchy_sketches_filenames': DATA_SOURCES['sketchy']['sketches'],
    'sketchy_test_photos': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy_test_photos_triplets': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy_test_photos_filenames': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy_test_sketches': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy_test_sketches_triplets': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy_test_sketches_filenames': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy_named_photos': DATA_SOURCES['sketchy']['photos'],
    'sketchy_named_sketches': DATA_SOURCES['sketchy']['photos'],
    'sketchy_test_named_photos': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy_test_named_sketches': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy_mixed_batches': 'sketchy',
    'sketchy_test_mixed_batches': 'sketchy_test'
}


DATASETS = {
    'sketchy_photos': lambda data_source: Sketchy(data_source),
    'sketchy_photos_triplets': lambda data_source: SketchyTriplets(data_source),
    'sketchy_photos_filenames': lambda data_source: SketchyFilenameIndexed(data_source),
    'sketchy_sketches': lambda data_source: Sketchy(data_source),
    'sketchy_sketches_triplets': lambda data_source: SketchyTriplets(data_source),
    'sketchy_sketches_filenames': lambda data_source: SketchyFilenameIndexed(data_source),
    'sketchy_test_photos': lambda data_source: Sketchy(data_source),
    'sketchy_test_photos_triplets': lambda data_source: SketchyTriplets(data_source),
    'sketchy_test_photos_filenames': lambda data_source: SketchyFilenameIndexed(data_source),
    'sketchy_test_sketches': lambda data_source: Sketchy(data_source),
    'sketchy_test_sketches_triplets': lambda data_source: SketchyTriplets(data_source),
    'sketchy_test_sketches_filenames': lambda data_source: SketchyFilenameIndexed(data_source),
    'sketchy_named_photos': lambda data_source: SketchyImageNames(data_source),
    'sketchy_named_sketches': lambda data_source: SketchyImageNames(data_source),
    'sketchy_test_named_photos': lambda data_source: SketchyImageNames(data_source),
    'sketchy_test_named_sketches': lambda data_source: SketchyImageNames(data_source),
    'sketchy_mixed_batches': lambda data_source: SketchyMixedBatches(data_source),
    'sketchy_test_mixed_batches': lambda data_source: SketchyMixedBatches(data_source)
}


def get_dataset(dataset_name):
    """
    Get the Dataset instancing lambda from the dictionary and return it's evaluation. This way, a Dataset object is
    only instanced when this function is evaluated.
    :param dataset_name: the name of the Dataset to be instanced. Must be a key in the DATASETS dictionary.
    :type: str
    :return: the corresponding Dataset object.
    :type: torch.utils.data.Dataset

    """
    try:
        return DATASETS[dataset_name](DATASET_DATA_SOURCES[dataset_name])
    except KeyError as e:
        raise type(e)('%s is not registered a Dataset.' % dataset_name)


def get_dataset_class_names(dataset_name):
    """
    Get a list of class names for the given dataset, indexed in alphabetical order.
    :param dataset_name:  the name of the dataset
    :type: str
    :return: a list of class names in alphabetical order
    :type: list<str>
    """
    import os
    return os.listdir(DATASET_DATA_SOURCES[dataset_name])
