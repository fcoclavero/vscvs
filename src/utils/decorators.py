__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" General utility decorators. """


import functools
import warnings

from threading import Thread


def deprecated(func):
    """
    Decorator that can be used to mark functions as deprecated, emitting a warning when the function is used.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which emits a warning when used
    :type: function
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def threaded(func):
    """
    Decorator that runs the decorated function asynchronously by throwing a new Python thread upon function call.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which evaluates in a new Python thread
    :type: function
    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        t = Thread(target=func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator


def parametrized(decorator):
    """
    Meta-decorator that adds parametrization support to other decorators.
    :param decorator: the decorator to be modified with parameter support
    :type: function
    :return: a trainer engine with the update function
    :type: function
    """
    @functools.wraps(decorator)
    def wrapper(*args, **kwargs):
        def decorator_wrapper(func):
            return decorator(func, *args, **kwargs)
        return decorator_wrapper
    return wrapper


@parametrized
def handle_exception_decorator(func, callback):
    """
    Method decorator that tries to execute the given function, and handles any exception by setting the
    object's status.
    :param func: the function to be decorated
    :type: callable
    :param callback: function to execute on exception. Receives the instance and the error.
    :type: callable
    :return: the decorated function
    :type: callable
    """
    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        try:
            return func(instance, *args, **kwargs)
        except Exception as e:
            callback(instance, func, e, *args, **kwargs)
    return wrapper
