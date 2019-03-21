import os, pickle, random

import pandas as pd

from settings import ROOT_DIR


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
    pickle.dump( # save binary labeled data to the static dir
        data,
        open(os.path.join(ROOT_DIR, 'static', 'pickles', 'discriminators', 'sample-vectors.pickle'),'wb')
    )
    data['class'] = data['class'].apply(lambda c: [1 - c, c]) # create one-hot encoded labels from the binary labels
    pickle.dump( # save one-hot encoding labeled data to the static dir
        data,
        open(os.path.join(ROOT_DIR, 'static', 'pickles', 'discriminators', 'sample-vectors-onehot.pickle'), 'wb')
    )