__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Click groups for different optimizers. """


import click

from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.trainers.mixins import AdamOptimizerMixin, SGDOptimizerMixin


@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--beta-1', prompt='Beta 1', default=.9,
              help='First coefficient used for computing running averages of gradient and its square.')
@click.option('--beta-2', prompt='Beta 2', default=.999,
              help='Second coefficient used for computing running averages of gradient and its square.')
@click.option('--epsilon', prompt='Epsilon', default=1e-8,
              help='Term added to the denominator to improve numerical stability .')
@click.option('--weight-decay', prompt='Weight decay', default=0.0, help='Weight decay (L2 penalty).')
@click.option('--amsgrad', prompt='Amsgrad', default=False, help='Whether to use the AMSGrad variant.')
@pass_kwargs_to_context
def adam(context, **kwargs):
    """ Train models using an Adam optimizer. """
    context.obj['betas'] = (context.obj.pop('beta_1'), context.obj.pop('beta_2'))
    context.obj['optimizer_mixin'] = AdamOptimizerMixin


@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--momentum', prompt='Momentum', help='Momentum parameter for SGD optimizer.', default=.2)
@pass_kwargs_to_context
def sgd(context, **kwargs):
    """ Train models using an SGD optimizer. """
    context.obj['optimizer_mixin'] = SGDOptimizerMixin
