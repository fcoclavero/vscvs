__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Register datasets here to make them available in the CLI. """


from vscvs.datasets.multimodal import SiameseDataset
from vscvs.datasets.sketchy import *


DATASETS = {
    'sketchy-photos':
        lambda *args, **kwargs: Sketchy('sketchy-photos', *args, **kwargs),
    'sketchy-photos-triplets':
        lambda *args, **kwargs: SketchyTriplets('sketchy-photos', *args, **kwargs),
    'sketchy-photos-filenames':
        lambda *args, **kwargs: SketchyFilenameIndexed('sketchy-photos', *args, **kwargs),
    'sketchy-photos-binary':
        lambda *args, **kwargs: SketchyBinaryEncoded('sketchy-photos', *args, **kwargs),
    'sketchy-photos-one-hot':
        lambda *args, **kwargs: SketchyOneHotEncoded('sketchy-photos', *args, **kwargs),
    'sketchy-sketches':
        lambda *args, **kwargs: Sketchy('sketchy-sketches', *args, **kwargs),
    'sketchy-sketches-triplets':
        lambda *args, **kwargs: SketchyTriplets('sketchy-sketches', *args, **kwargs),
    'sketchy-sketches-filenames':
        lambda *args, **kwargs: SketchyFilenameIndexed('sketchy-sketches', *args, **kwargs),
    'sketchy-sketches-binary':
        lambda *args, **kwargs: SketchyBinaryEncoded('sketchy-sketches', *args, **kwargs),
    'sketchy-sketches-one-hot':
        lambda *args, **kwargs: SketchyOneHotEncoded('sketchy-sketches', *args, **kwargs),
    'sketchy-test-photos':
        lambda *args, **kwargs: Sketchy('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-photos-triplets':
        lambda *args, **kwargs: SketchyTriplets('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-photos-filenames':
        lambda *args, **kwargs: SketchyFilenameIndexed('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-photos-binary':
        lambda *args, **kwargs: SketchyBinaryEncoded('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-photos-one-hot':
        lambda *args, **kwargs: SketchyOneHotEncoded('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-sketches':
        lambda *args, **kwargs: Sketchy('sketchy-test-sketches', *args, **kwargs),
    'sketchy-test-sketches-triplets':
        lambda *args, **kwargs: SketchyTriplets('sketchy-test-sketches', *args, **kwargs),
    'sketchy-test-sketches-filenames':
        lambda *args, **kwargs: SketchyFilenameIndexed('sketchy-test-sketches', *args, **kwargs),
    'sketchy-test-sketches-binary':
        lambda *args, **kwargs: SketchyBinaryEncoded('sketchy-test-sketches', *args, **kwargs),
    'sketchy-test-sketches-one-hot':
        lambda *args, **kwargs: SketchyOneHotEncoded('sketchy-test-sketches', *args, **kwargs),
    'sketchy-named-photos':
        lambda *args, **kwargs: SketchyImageNames('sketchy-photos', *args, **kwargs),
    'sketchy-named-sketches':
        lambda *args, **kwargs: SketchyImageNames('sketchy-sketches', *args, **kwargs),
    'sketchy-test-named-photos':
        lambda *args, **kwargs: SketchyImageNames('sketchy-test-photos', *args, **kwargs),
    'sketchy-test-named-sketches':
        lambda *args, **kwargs: SketchyImageNames('sketchy-test-sketches', *args, **kwargs),
    'sketchy-siamese':
        lambda *args, **kwargs: SiameseDataset(
            Sketchy('sketchy-photos', *args, **kwargs),
            Sketchy('sketchy-sketches', *args, **kwargs)
        ),
    'sketchy-test-siamese':
        lambda *args, **kwargs: SiameseDataset(
            Sketchy('sketchy-test-photos', *args, **kwargs),
            Sketchy('sketchy-test-sketches', *args, **kwargs)
        ),
    'sketchy-mixed-batches':
        lambda *args, **kwargs: SketchyMixedBatches('sketchy-photos', 'sketchy-sketches', *args, **kwargs),
    'sketchy-test-mixed-batches':
        lambda *args, **kwargs: SketchyMixedBatches('sketchy-test-photos', *args, **kwargs),
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
        return DATASETS[dataset_name](*args, **kwargs)
    except KeyError as e:
        raise type(e)('{} is not registered a Dataset.'.format(dataset_name))
