__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Launch Paperspace Gradient jobs using the Python CLI. """


import os
import sys

import click

from dotenv import dotenv_values
from gradient import ExperimentsClient

from vscvs.utils import camel_to_snake_case_dict_keys
from vscvs.utils import load_yaml


@click.group()
@click.option("--name", prompt="Job name", help="Job name.", default="New job")
def gradient(name):
    """ Launch jobs in the Paperspace cloud using the Gradient SDK. """
    print("Running command in Gradient.")
    experiment_client = ExperimentsClient(api_key=os.environ["GRADIENT_API_KEY"])
    command_arguments = sys.argv[4:]  # skipped elements are the script name, the gradient command and its options
    experiment_parameters = {
        "name": name,
        "experiment_env": dotenv_values(".env.gradient"),
        "command": "python -m vscvs {}".format(" ".join(command_arguments)),
        **camel_to_snake_case_dict_keys(load_yaml("gradient.yaml")),
    }  # shared experiment parameters, such as machine type
    experiment_id = experiment_client.run_single_node(**experiment_parameters)  # define and run experiment
    stream_logs(experiment_id)
    sys.exit()  # terminate program after log stream


@click.command()
def dropbox_upload():
    """ Upload Paperspace storage to Dropbox. """
    print("Uploading to DBX.")
    experiment_client = ExperimentsClient(api_key=os.environ["GRADIENT_API_KEY"])
    experiment_parameters = {
        "name": "dropbox-upload",
        "experiment_env": dotenv_values(".env.gradient"),
        "command": "ls -la /storage/other && cd /storage/other && find /storage/vscvs/data | while read file; do echo $file; target=Workspace/Python/Tesis/paperspace/$file; ./dbxcli put $file $target; done",
        **camel_to_snake_case_dict_keys(load_yaml("gradient.yaml")),
    }  # shared experiment parameters, such as machine type
    experiment_id = experiment_client.run_single_node(**experiment_parameters)  # define and run experiment
    stream_logs(experiment_id)
    sys.exit()  # terminate program after log stream


def stream_logs(experiment_id):
    """
    Stream the logs of a Gradient experiment to the local I/O.
    :param experiment_id: id of the experiments who's logs are to be streamed.
    :type: str
    """
    experiment_client = ExperimentsClient(api_key=os.environ["GRADIENT_API_KEY"])
    log_stream = experiment_client.yield_logs(experiment_id)
    print("Streaming logs of experiment {}.".format(experiment_id))
    # noinspection PyBroadException
    try:
        while True:
            print(log_stream.send(None).message)
    except:
        print("Log stream ended.")
