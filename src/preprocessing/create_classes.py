__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Functions to generate a class dictionary for image labels. These are preprocessed and embedded (FastText). """


import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from plotly.offline import plot
from sklearn.manifold import TSNE

from settings import DATA_SOURCES

from modules.textpreprocess.compound_cleaners.en import full_clean
from modules.wordvectors.en import document_vector


def classes_set(directory):
    """
    Returns a set containing all classes in a given data set. Data sets must be structured in the following fashion:
    /data_set_name/<photos><sketches>/<classes>. In other words, they must have a root folder, inside which must be
    a photos and a sketches folder, each containing a folder for each class. Class names are inferred from the names
    of these directories.
    :param directory: path to either an image or sketch folder
    :type: str
    :return: a set containing all class names
    :type: str
    """
    return set([path for path in os.listdir(directory) if os.path.isdir(os.path.join(directory, path))])


def plot_classes(classes):
    """
    Plot a classes data frame, showing all the different classes in a 2D projection of the semantic space.
    :param classes: a classes data frame, with "class", "vector" and "tsne" columns
    :type: pd.DataFrame
    :return: None
    """
    aux = np.vstack(classes['tsne'])
    trace = go.Scattergl(
        x=aux[:, 0],
        y=aux[:, 1],
        text=classes['class'].values,
        mode='markers',
        marker=dict(
            size=16,
            color=np.random.randn(len(aux)),
            colorscale='Viridis'
        )
    )
    data = [trace]
    plot(data)


def create_classes_data_frame(data_set, tsne_dimension = 2):
    """
    Create a new classes data frame for the specified dataset. The dataset must be registered in the project settings.
    The data frame is pickled before function return, to prevent re-calculating things.
    :param data_set: the name of the dataset
    :type: str
    :param tsne_dimension: the dimensions for the lower dimensional vector projections
    :type: int
    :return: a pandas DataFrame with "class", "vector" (document embeddings) and "tsne" columns
    :type: pd.DataFrame
    """
    paths = classes_set(DATA_SOURCES[data_set]['photos']).union(classes_set(DATA_SOURCES[data_set]['sketches']))
    classes = pd.DataFrame(columns=['class', 'vector', 'tsne'])
    classes['class'] = sorted(list(paths))
    classes['class'] = classes['class'].apply(lambda cls: ' '.join(re.split(r'(?:_|-)', cls)))
    classes['class'] = classes['class'].apply(full_clean)
    classes['vector'] = classes['class'].apply(document_vector)
    classes['tsne'] = list(TSNE(n_components=tsne_dimension).fit_transform(np.vstack(classes['vector'].values)))
    pickle.dump(classes, open(DATA_SOURCES[data_set]['classes'], 'wb'))
    return classes
