__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Path handler. """


import os
import shutil

from datetime import datetime

from settings import ROOT_DIR, CHECKPOINT_NAME_FORMAT


def get_cache_directory(cache_filename):
    """
    Get the path where a cache file should be stored.
    :param cache_filename: the name of the cache file.
    :type: str
    :return: the model checkpoint path.
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'cache', cache_filename)


def get_checkpoint_directory(model_name, tag=None, date=datetime.now()):
    """
    Get the path where model checkpoints should be stored.
    :param model_name: the name of the model
    :type: str
    :param tag: optional tag for model checkpoint and tensorboard logs.
    :type: str
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'checkpoints', model_name, tag or '', date.strftime(CHECKPOINT_NAME_FORMAT))


def get_embedding_directory(embedding_folder_name, tags=None):
    """
    Get the path where tensorboard embeddings should be stored.
    :param embedding_folder_name: the name of the folder which will contain the embeddings.
    :type: str
    :param tags: optional tags organizing embeddings.
    :type: List[str]
    :return: the image path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'logs', 'embeddings', embedding_folder_name, *tags or '')


def get_image_directory(image_folder_name, tags=None):
    """
    Get the path where tensorboard images should be stored.
    :param image_folder_name: the name of the folder which will contain the images.
    :type: str
    :param tags: optional tags organizing images.
    :type: List[str]
    :return: the image path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'logs', 'images', image_folder_name, *tags or '')


def get_log_directory(model_name, tag=None, date=datetime.now()):
    """
    Get the path where model checkpoints should be stored.
    :param model_name: the name of the model.
    :type: str
    :param tag: optional tag for model checkpoint and tensorboard logs.
    :type: str
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'logs', model_name, tag or '', date.strftime(CHECKPOINT_NAME_FORMAT))


def get_subdirectories(path):
    """
    Get a list of all the child directories of the given path.
    :param path: the path who's child directories are to be returned.
    :type: str
    :return: the paths of the child directories, relative to the given path.
    :type: List[str]
    """
    return next(os.walk(path))[1]


def recreate_directory(directory_path):
    """
    Delete and recreate the directory at the given path to ensure an empty directory.
    :param directory_path: the path to the directory to be recreated.
    :type: str
    """
    shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path)