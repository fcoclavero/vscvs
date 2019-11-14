__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Register datasets here to make them available in the CLI. """


import os
import pickle

from settings import DATA_SOURCES
from src.datasets.sketchy import Sketchy, SketchyImageNames, SketchyMixedBatches, SketchyTriplets, \
                                 SketchyFilenameIndexed, SketchyBinaryEncoded


DATASET_DATA_SOURCES = {
    'sketchy-photos': DATA_SOURCES['sketchy']['photos'],
    'sketchy-photos-triplets': DATA_SOURCES['sketchy']['photos'],
    'sketchy-photos-filenames': DATA_SOURCES['sketchy']['photos'],
    'sketchy-photos-binary': DATA_SOURCES['sketchy']['photos'],
    'sketchy-sketches': DATA_SOURCES['sketchy']['sketches'],
    'sketchy-sketches-triplets': DATA_SOURCES['sketchy']['sketches'],
    'sketchy-sketches-filenames': DATA_SOURCES['sketchy']['sketches'],
    'sketchy-sketches-binary': DATA_SOURCES['sketchy']['sketches'],
    'sketchy-test-photos': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-test-photos-triplets': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-test-photos-filenames': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-test-photos-binary': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-test-sketches': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy-test-sketches-triplets': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy-test-sketches-filenames': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy-test-sketches-binary': DATA_SOURCES['sketchy_test']['sketches'],
    'sketchy-named-photos': DATA_SOURCES['sketchy']['photos'],
    'sketchy-named-sketches': DATA_SOURCES['sketchy']['photos'],
    'sketchy-test-named-photos': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-test-named-sketches': DATA_SOURCES['sketchy_test']['photos'],
    'sketchy-mixed-batches': 'sketchy',
    'sketchy-test-mixed-batches': 'sketchy_test',
}


DATASETS = {
    'sketchy-photos':
        lambda data_source, *args, **kwargs: Sketchy(data_source, *args, **kwargs),
    'sketchy-photos-triplets':
        lambda data_source, *args, **kwargs: SketchyTriplets(data_source, *args, **kwargs),
    'sketchy-photos-filenames':
        lambda data_source, *args, **kwargs: SketchyFilenameIndexed(data_source, *args, **kwargs),
    'sketchy-photos-binary':
        lambda data_source, *args, **kwargs: SketchyBinaryEncoded(data_source, *args, **kwargs),
    'sketchy-sketches':
        lambda data_source, *args, **kwargs: Sketchy(data_source, *args, **kwargs),
    'sketchy-sketches-triplets':
        lambda data_source, *args, **kwargs: SketchyTriplets(data_source, *args, **kwargs),
    'sketchy-sketches-filenames':
        lambda data_source, *args, **kwargs: SketchyFilenameIndexed(data_source, *args, **kwargs),
    'sketchy-sketches-binary':
        lambda data_source, *args, **kwargs: SketchyBinaryEncoded(data_source, *args, **kwargs),
    'sketchy-test-photos':
        lambda data_source, *args, **kwargs: Sketchy(data_source, *args, **kwargs),
    'sketchy-test-photos-triplets':
        lambda data_source, *args, **kwargs: SketchyTriplets(data_source, *args, **kwargs),
    'sketchy-test-photos-filenames':
        lambda data_source, *args, **kwargs: SketchyFilenameIndexed(data_source, *args, **kwargs),
    'sketchy-test-photos-binary':
        lambda data_source, *args, **kwargs: SketchyBinaryEncoded(data_source, *args, **kwargs),
    'sketchy-test-sketches':
        lambda data_source, *args, **kwargs: Sketchy(data_source, *args, **kwargs),
    'sketchy-test-sketches-triplets':
        lambda data_source, *args, **kwargs: SketchyTriplets(data_source, *args, **kwargs),
    'sketchy-test-sketches-filenames':
        lambda data_source, *args, **kwargs: SketchyFilenameIndexed(data_source, *args, **kwargs),
    'sketchy-test-sketches-binary':
        lambda data_source, *args, **kwargs: SketchyBinaryEncoded(data_source, *args, **kwargs),
    'sketchy-named-photos':
        lambda data_source, *args, **kwargs: SketchyImageNames(data_source, *args, **kwargs),
    'sketchy-named-sketches':
        lambda data_source, *args, **kwargs: SketchyImageNames(data_source, *args, **kwargs),
    'sketchy-test-named-photos':
        lambda data_source, *args, **kwargs: SketchyImageNames(data_source, *args, **kwargs),
    'sketchy-test-named-sketches':
        lambda data_source, *args, **kwargs: SketchyImageNames(data_source, *args, **kwargs),
    'sketchy-mixed-batches':
        lambda data_source, *args, **kwargs: SketchyMixedBatches(data_source, *args, **kwargs),
    'sketchy-test-mixed-batches':
        lambda data_source, *args, **kwargs: SketchyMixedBatches(data_source, *args, **kwargs),
}


def get_dataset(dataset_name, *args, **kwargs):
    """
    Get the Dataset instancing lambda from the dictionary and return it's evaluation. This way, a Dataset object is
    only instanced when this function is evaluated.
    :param dataset_name: the name of the Dataset to be instanced. Must be a key in the DATASETS dictionary.
    :type: str
    :return: the corresponding Dataset object.
    :type: torch.utils.data.Dataset
    """
    try:
        return DATASETS[dataset_name](DATASET_DATA_SOURCES[dataset_name], *args, **kwargs)
    except KeyError as e:
        raise type(e)('{} is not registered a Dataset.'.format(dataset_name))


def get_dataset_classes_dataframe(dataset_name):
    """
    Return the dataset's classes dataframe, which includes class names and class word vectors.
    :param dataset_name: the name of the Dataset. Must be a key in the DATASETS dictionary.
    :type: str
    :return: the corresponding classes dataframe
    :type: pandas.Dataframes
    """
    try:
        return pickle.load(open(os.path.join(DATASET_DATA_SOURCES[dataset_name], 'classes.pickle'), 'rb'))
    except KeyError as e:
        raise type(e)('{} doest not have a classes dataframe.'.format(dataset_name))
