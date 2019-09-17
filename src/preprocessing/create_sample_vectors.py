__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Function for creating a dataset with random vectors, for testing purposes. """


import pickle, random

import pandas as pd

from settings import DATA_SOURCES


def create_sample_vectors(n, dimension):
    """
    Create a sample vector dataframe that can be loaded as a dataset for discriminator tests. Sample vectors are
    pickled in the static directory.
    :param n: the number of samples to be created
    :type: int
    :param dimension: the dimensionality for the sample vectors
    :type: int
    """
    data = pd.DataFrame(columns=['class', 'vector'])
    data['class'] = [random.randint(0, 1) for i in range(n)]
    data['vector'] = data['class'].apply(lambda c: [c + random.uniform(0, 1) for i in range(dimension)])
    pickle.dump(data, open(DATA_SOURCES['sample_vectors']['pickle']), 'wb') # save binary labeled data to the static dir
    data['class'] = data['class'].apply(lambda c: [1 - c, c]) # create one-hot encoded labels from the binary labels
    pickle.dump(data, open(DATA_SOURCES['sample_vectors_onehot']['pickle'], 'wb')) # save to the static dir
    DATA_SOURCES['sample_vectors']['dimensions'] = (n, dimension) # Update dimensions
    DATA_SOURCES['sample_vectors_onehot']['dimensions'] = (n, dimension)