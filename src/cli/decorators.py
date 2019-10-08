__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Decorators for the click CLI. """


import click
import functools


def pass_args_as_context(func):
    """
    Decorator for `click` CLIs that puts all the kwargs of the decorated function to the click context and passes it
    on with `click.pass_context`. This is useful when multiple commands receive the same objects. In this case, a click
    group can be created and decorated with this decorator, enabling the shared parameters to be received by the click
    group and then be passed on to the different commands.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which passes all kwargs to the click context and passes it on
    :type: function
    """
    @click.pass_context # pass context decorator is needed for the variables to be accessible in the decorated function
    def new_func(context, *args, **kwargs):
        # Extend the context dict with the provided kwargs
        context.obj = {**context.obj, **kwargs} if context.obj else kwargs
        return context.invoke(func, context.obj, *args, **kwargs)
    return functools.update_wrapper(new_func, func)
