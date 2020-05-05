__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Click groups for different optimizers. """


import click

from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.trainers.mixins import AdaBoundOptimizerMixin, AdamOptimizerMixin, AdamWOptimizerMixin, \
    RMSpropOptimizerMixin, SGDOptimizerMixin


# noinspection DuplicatedCode
@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=1e-3)
@click.option('--beta-1', prompt='Beta 1', default=.9,
              help='First coefficient used for computing running averages of gradient and its square.')
@click.option('--beta-2', prompt='Beta 2', default=.999,
              help='Second coefficient used for computing running averages of gradient and its square.')
@click.option('--final-learning-rate', prompt='Final learning rate', help='Final (SGD) learning rate.', default=.1)
@click.option('--gamma', prompt='Gamma', help='Convergence speed of the bound functions.', default=1e-3)
@click.option('--epsilon', prompt='Epsilon', default=1e-8,
              help='Term added to the denominator to improve numerical stability .')
@click.option('--weight-decay', prompt='Weight decay', default=0.0, help='Weight decay (L2 penalty).')
@click.option('--amsbound', prompt='AmsBound', default=False, help='Whether to use the AMSBound variant.')
@pass_kwargs_to_context
def adabound(context, **__):
    """ Train models using an AdaBound optimizer. """
    context.obj['betas'] = (context.obj.pop('beta_1'), context.obj.pop('beta_2'))
    context.obj['optimizer_mixin'] = AdaBoundOptimizerMixin


# noinspection DuplicatedCode
@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=1e-3)
@click.option('--beta-1', prompt='Beta 1', default=.9,
              help='First coefficient used for computing running averages of gradient and its square.')
@click.option('--beta-2', prompt='Beta 2', default=.999,
              help='Second coefficient used for computing running averages of gradient and its square.')
@click.option('--epsilon', prompt='Epsilon', default=1e-8,
              help='Term added to the denominator to improve numerical stability .')
@click.option('--weight-decay', prompt='Weight decay', default=0.0, help='Weight decay (L2 penalty).')
@click.option('--amsgrad', prompt='Amsgrad', default=False, help='Whether to use the AMSGrad variant.')
@pass_kwargs_to_context
def adam(context, **__):
    """ Train models using an Adam optimizer. """
    context.obj['betas'] = (context.obj.pop('beta_1'), context.obj.pop('beta_2'))
    context.obj['optimizer_mixin'] = AdamOptimizerMixin


# noinspection DuplicatedCode
@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=1e-3)
@click.option('--beta-1', prompt='Beta 1', default=.9,
              help='First coefficient used for computing running averages of gradient and its square.')
@click.option('--beta-2', prompt='Beta 2', default=.999,
              help='Second coefficient used for computing running averages of gradient and its square.')
@click.option('--epsilon', prompt='Epsilon', default=1e-8,
              help='Term added to the denominator to improve numerical stability .')
@click.option('--weight-decay', prompt='Weight decay', default=0.0, help='Weight decay (L2 penalty).')
@click.option('--amsgrad', prompt='Amsgrad', default=False, help='Whether to use the AMSGrad variant.')
@pass_kwargs_to_context
def adam_w(context, **__):
    """ Train models using an AdamW optimizer. """
    context.obj['betas'] = (context.obj.pop('beta_1'), context.obj.pop('beta_2'))
    context.obj['optimizer_mixin'] = AdamWOptimizerMixin


# noinspection DuplicatedCode
@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=1e-3)
@click.option('--alpha', prompt='Alpha', default=.99, help='Smoothing constant.')
@click.option('--epsilon', prompt='Epsilon', default=1e-8,
              help='Term added to the denominator to improve numerical stability .')
@click.option('--weight-decay', prompt='Weight decay', default=0.0, help='Weight decay (L2 penalty).')
@click.option('--momentum', prompt='Momentum', help='Momentum factor.', default=0.0)
@click.option('--centered', prompt='Amsgrad', default=False,
              help='whether to compute the centered RMSProp (gradient normalized by an estimation of its variance).')
@pass_kwargs_to_context
def rms_prop(context, **__):
    """ Train models using an RMSProp optimizer. """
    context.obj['optimizer_mixin'] = RMSpropOptimizerMixin


# noinspection DuplicatedCode
@click.group()
@click.option('--learning-rate', prompt='Learning rate', help='Learning rate for the optimizer', default=1e-3)
@click.option('--momentum', prompt='Momentum', help='Momentum parameter for SGD optimizer.', default=.2)
@pass_kwargs_to_context
def sgd(context, **__):
    """ Train models using an SGD optimizer. """
    context.obj['optimizer_mixin'] = SGDOptimizerMixin
