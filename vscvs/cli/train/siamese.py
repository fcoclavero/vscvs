__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Siamese model training entry point. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs
from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.loss_functions import ReductionMixin
from vscvs.utils import load_classification_model_from_checkpoint


@click.group()
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the dataset to be used for training.",
    type=click.Choice(["sketchy", "sketchy-test"]),
)
@click.option(
    "--loss-reduction",
    prompt="Loss reduction",
    help="Reduction function for the loss function.",
    type=click.Choice(ReductionMixin.reduction_choices),
)
@click.option("--margin", prompt="Margin", help="The margin for the contrastive loss.", default=0.2)
@pass_kwargs_to_context
def siamese(context, *_, **__):
    """ Train a siamese model. """
    context.obj["dataset_name"] = context.obj["dataset_name"] + "-siamese"


@click.group()
@click.option(
    "--first-branch-checkpoint",
    prompt="First branch checkpoint name",
    help="Name of the checkpoint directory for the first branch.",
)
@click.option(
    "--first-branch-date",
    prompt="First branch checkpoint date",
    help="Checkpoint date (corresponds to the directory name) for the first branch.",
)
@click.option(
    "--first-branch-state-dict",
    prompt="First branch state dict",
    help="The state_dict file to be loaded for the first branch.",
)
@click.option(
    "-tf",
    "--first-branch-tag",
    help="Optional tag for first branch model checkpoint and tensorboard logs.",
    multiple=True,
)
@click.option(
    "--second-branch-checkpoint",
    prompt="Second branch checkpoint name",
    help="Name of the checkpoint directory for the second branch.",
)
@click.option(
    "--second-branch-date",
    prompt="Second branch checkpoint date",
    help="Checkpoint date (corresponds to the directory name) for the second branch.",
)
@click.option(
    "--second-branch-state-dict",
    prompt="Second branch state dict",
    help="The state_dict file to be loaded for the second branch.",
)
@click.option(
    "-ts",
    "--second-branch-tag",
    help="Optional tag for second branch model checkpoint and tensorboard logs.",
    multiple=True,
)
@pass_kwargs_to_context
def pretrained(
    context,
    first_branch_checkpoint,
    first_branch_date,
    first_branch_state_dict,
    first_branch_tag,
    second_branch_checkpoint,
    second_branch_date,
    second_branch_state_dict,
    second_branch_tag,
    *_,
    **__
):
    from vscvs.models import ResNext

    if first_branch_checkpoint and first_branch_date and first_branch_state_dict:
        context.obj.pop("first_branch_checkpoint")
        context.obj.pop("first_branch_date")
        context.obj.pop("first_branch_state_dict")
        context.obj.pop("first_branch_tag")
        context.obj["embedding_network_0"] = load_classification_model_from_checkpoint(
            ResNext, first_branch_state_dict, first_branch_checkpoint, first_branch_date, *first_branch_tag
        )
    if second_branch_checkpoint and second_branch_date and second_branch_state_dict:
        context.obj.pop("second_branch_checkpoint")
        context.obj.pop("second_branch_date")
        context.obj.pop("second_branch_state_dict")
        context.obj.pop("second_branch_tag")
        context.obj["embedding_network_1"] = load_classification_model_from_checkpoint(
            ResNext, second_branch_state_dict, second_branch_checkpoint, second_branch_date, *second_branch_tag
        )


@click.command()
@pass_context_to_kwargs
def cnn(*args, **kwargs):
    """ Train a siamese CNN model. """
    from vscvs.trainers.siamese import train_siamese_cnn

    click.echo("siamese cnn - {} dataset".format(kwargs["dataset_name"]))
    train_siamese_cnn(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
def resnet(*args, **kwargs):
    """ Train a siamese ResNet model. """
    from vscvs.trainers.siamese import train_siamese_resnet

    click.echo("siamese resnet - {} dataset".format(kwargs["dataset_name"]))
    train_siamese_resnet(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
def resnext(*args, **kwargs):
    """ Train a siamese ResNext model. """
    from vscvs.trainers.siamese import train_siamese_resnext

    click.echo("siamese resnext - {} dataset".format(kwargs["dataset_name"]))
    train_siamese_resnext(*args, **kwargs)


for command in [cnn, resnet, resnext]:
    pretrained.add_command(command)
    siamese.add_command(command)


siamese.add_command(pretrained)
