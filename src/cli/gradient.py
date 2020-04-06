__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Launch Paperspace Gradient jobs using the Python CLI. """


import click
import os
import sys

from dotenv import dotenv_values
from gradient import ExperimentsClient

from src.utils import camel_to_snake_case_dict_keys, load_yaml


@click.group()
@click.option('--name', prompt='Job name', help='Job name.', default='New job')
def gradient(name):
    """ Launch jobs in the Paperspace cloud using the Gradient SDK. """
    print('Running command in Gradient.')
    experiment_client = ExperimentsClient(api_key=os.environ['GRADIENT_API_KEY'])
    command_arguments = sys.argv[4:]  # first elements in `sys.argv` are the script name, the gradient command and it's options
    experiment_parameters = {
        'name': name,
        'experiment_env': dotenv_values('.env.gradient'),
        'command': 'python main.py {}'.format(' '.join(command_arguments)),
        **camel_to_snake_case_dict_keys(load_yaml('gradient.yaml')) # shared experiment parameters, such as machine type
    }
    experiment_id = experiment_client.run_single_node(**experiment_parameters) # define and run experiment
    stream_logs(experiment_id)
    sys.exit() # terminate program after log stream


def stream_logs(experiment_id):
    """
    Stream the logs of a Gradient experiment to the local I/O.
    :param experiment_id: id of the experiments who's logs are to be streamed.
    :type: str
    """
    experiment_client = ExperimentsClient(api_key=os.environ['GRADIENT_API_KEY'])
    log_stream = experiment_client.yield_logs(experiment_id)
    print('Streaming logs of experiment {}.'.format(experiment_id))
    try:
        while True:
            print(log_stream.send(None).message)
    except:
        print('Log stream ended.')
