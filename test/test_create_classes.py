import os
import pickle

import numpy as np

from settings import DATA_SETS, ROOT_DIR

from src.preprocessing.create_classes import classes_set, create_classes_data_frame


def test_classes_set():
    """
    Check that the test data set class names are correctly identified.
    """
    assert classes_set(DATA_SETS['sketchy_test']['images']) == pickle.load(
        open(os.path.join(ROOT_DIR, 'static\\pickles\\classes_set.pickle'), 'rb')
    )


def test_create_classes_data_frame():
    """
    Check that the class name object is correctly created, containing the correct class names and vectors.
    TSNE projections are omitted, as they change every time they are calculated.
    :return:
    """
    try:
        os.remove(DATA_SETS['sketchy_test']['classes'])
    except OSError:
        pass
    test_classes = pickle.load(open(os.path.join(ROOT_DIR,'static\\pickles\\classes.pickle'), 'rb'))
    new_classes = create_classes_data_frame('sketchy_test')
    assert test_classes['class'].equals(new_classes['class'])
    for i, vector in enumerate(test_classes['vector']):
        np.testing.assert_almost_equal(vector, new_classes['vector'][i])
