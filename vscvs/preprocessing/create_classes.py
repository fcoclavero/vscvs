__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Functions to generate a class dataframe for image labels. These are preprocessed and embedded (FastText). """


import os
import pickle
import re
import torch
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from plotly.offline import plot
from sklearn.manifold import TSNE
from torch.nn import PairwiseDistance, CosineSimilarity
from tqdm import tqdm

from modules.textpreprocess.compound_cleaners.en import full_clean
from modules.wordvectors.en import document_vector
from settings import DATA_SOURCES


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


def create_classes_data_frame(dataset_name, distance='cosine', tsne_dimension=2):
    """
    Create a new classes dataframe for the specified dataset. The dataset must be registered in the project settings.
    The data frame is pickled before function return, to prevent re-calculating things.
    :param dataset_name: the name of the dataset
    :type: str
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :param tsne_dimension: the dimensions for the lower dimensional vector projections
    :type: int
    :return: a pandas DataFrame with "class", "vector" (document embeddings) and "tsne" columns
    :type: pd.DataFrame
    """
    dataset_dir = DATA_SOURCES[dataset_name]['images']
    paths = classes_set(dataset_dir)
    classes = pd.DataFrame(columns=['class', 'vector', 'tsne'])
    classes['classes'] = sorted(list(paths))
    tqdm.pandas(desc='Removing special characters.')
    classes['classes'] = classes['classes'].progress_apply(lambda cls: ' '.join(re.split(r'(?:_|-)', cls)))
    tqdm.pandas(desc='Applying full clean.')
    classes['classes'] = classes['classes'].progress_apply(full_clean)
    tqdm.pandas(desc='Creating document vectors.')
    vectors = torch.tensor(np.vstack(classes['classes'].progress_apply(document_vector)))
    classes['vectors'] = vectors
    p_dist = PairwiseDistance(p=2) if distance == 'pairwise' else CosineSimilarity()
    classes['distances'] = p_dist( # distance from every node to every node
        vectors.repeat_interleave(vectors.shape[0], 0),  # each index repeated num_edges times
        vectors.repeat(vectors.shape[0], 1)  # the index range repeated num_edges times
    ).reshape(vectors.shape[0], -1) # convert to 2D matrix with shape [vectors.shape[0], vectors.shape[0]]
    classes['tsne'] = torch.tensor(TSNE(n_components=tsne_dimension).fit_transform(vectors))
    pickle.dump(classes, open(os.path.join(dataset_dir, 'classes.pickle'), 'wb'))
    return classes
