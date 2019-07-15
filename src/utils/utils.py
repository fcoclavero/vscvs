__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" General utilities. All functions are imported in the `utils.__init__.py` for ease of access. """


import os
import torch

from datetime import datetime

from settings import CHECKPOINT_NAME_FORMAT, ROOT_DIR


def get_device(n_gpu):
    """
    Returns the name of the PyTorch device to be used, based on the number of available gpus.
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :return: the name of the device to be used by PyTorch
    :type: str
    """
    return torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")


def get_checkpoint_directory(model_name, date=None):
    """
    Get the path where model checkpoints should be stored.
    :param model_name: the name of the model
    :type: str
    :param date: the date string of the model checkpoint. Defaults to the current date if `None` is passed.
    :type: str or None
    :return: the model checkpoint path
    :type: str
    """
    return os.path.join(
        ROOT_DIR, 'static', 'checkpoints', model_name, datetime.now().strftime(CHECKPOINT_NAME_FORMAT)
    )