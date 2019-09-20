__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Register datasets here to make them available in the CLI. """

from settings import DATA_SOURCES

from src.datasets.sketchy import Sketchy, SketchyImageNames, SketchyMixedBatches, SketchyTriplets, \
                                 SketchyFilenameIndexed

DATASETS = {
    'sketchy_photos': lambda: Sketchy(DATA_SOURCES['sketchy']['photos']),
    'sketchy_photos_triplets': lambda: SketchyTriplets(DATA_SOURCES['sketchy']['photos']),
    'sketchy_photos_filenames': lambda: SketchyFilenameIndexed(DATA_SOURCES['sketchy']['photos']),
    'sketchy_sketches': lambda: Sketchy(DATA_SOURCES['sketchy']['sketches']),
    'sketchy_sketches_triplets': lambda: SketchyTriplets(DATA_SOURCES['sketchy']['sketches']),
    'sketchy_sketches_filenames': lambda: SketchyFilenameIndexed(DATA_SOURCES['sketchy']['sketches']),
    'sketchy_test_photos': lambda: Sketchy(DATA_SOURCES['sketchy_test']['photos']),
    'sketchy_test_photos_triplets': lambda: SketchyTriplets(DATA_SOURCES['sketchy_test']['photos']),
    'sketchy_test_photos_filenames': lambda: SketchyFilenameIndexed(DATA_SOURCES['sketchy_test']['photos']),
    'sketchy_test_sketches': lambda: Sketchy(DATA_SOURCES['sketchy_test']['sketches']),
    'sketchy_test_sketches_triplets': lambda: SketchyTriplets(DATA_SOURCES['sketchy_test']['sketches']),
    'sketchy_test_sketches_filenames': lambda: SketchyFilenameIndexed(DATA_SOURCES['sketchy_test']['sketches']),
    'sketchy_named_photos': lambda: SketchyImageNames(DATA_SOURCES['sketchy']['photos']),
    'sketchy_named_sketches': lambda: SketchyImageNames(DATA_SOURCES['sketchy']['photos']),
    'sketchy_test_named_photos': lambda: SketchyImageNames(DATA_SOURCES['sketchy_test']['photos']),
    'sketchy_test_named_sketches': lambda: SketchyImageNames(DATA_SOURCES['sketchy_test']['photos']),
    'sketchy_mixed_batches': lambda: SketchyMixedBatches('sketchy'),
    'sketchy_test_mixed_batches': lambda: SketchyMixedBatches('sketchy_test')
}


def get_dataset(dataset_name):
    """
    Get the Dataset instancing lambda from the dictionary and return it's evaluation. This way, a Dataset object is
    only instanced when this function is evaluated.
    :param dataset_name: the name of the Dataset to be instanced. Must be a key in the DATASETS dictionary.
    :return:
    """
    try:
        return DATASETS[dataset_name]()
    except KeyError as e:
        raise type(e)('%s is not registered a Dataset.' % dataset_name)
