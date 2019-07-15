__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" General utilities. All functions are imported in the `utils.__init__.py` for ease of access. """


import torch


def get_device(n_gpu):
    """
    Returns the name of the PyTorch device to be used, based on the number of available gpus.
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    """
    return torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")