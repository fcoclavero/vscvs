__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Path handler. """


import os
import shutil

from datetime import datetime

from settings import CHECKPOINT_NAME_FORMAT
from settings import ROOT_DIR


def get_path(*paths):
    """
    Get the path of a file or directory by joining the given path components. Project files will be stored under the
    `data` subdirectory of the projects `ROOT_DIR`, which is set as an environment variable.
    :param paths: path components.
    :type: List[str]
    :return: the actual path.
    :type: str
    """
    return os.path.join(ROOT_DIR, "data", *paths or "")


def get_checkpoint_path(checkpoint_name, *tags, date=datetime.now()):
    """
    Get the path where trainer model checkpoints should be stored.
    :param checkpoint_name: the name of the model
    :type: str
    :param tags: optional tags for organizing checkpoints.
    :type: List[str]
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return get_path("checkpoints", checkpoint_name, *tags, date.strftime(CHECKPOINT_NAME_FORMAT))


def get_log_directory(model_name, *tags, date=datetime.now()):
    """
    Get the path where trainer tensorboard logs should be stored.
    :param model_name: the name of the model.
    :type: str
    :param tags: optional tags for organizing tensorboard logs.
    :type: List[str]
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return get_path("tensorboard", model_name, *tags, date.strftime(CHECKPOINT_NAME_FORMAT))


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
