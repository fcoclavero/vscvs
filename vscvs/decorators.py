__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" General decorators. """


import functools
import warnings

from datetime import datetime
from threading import Thread

import torch


def deprecated(func):
    """
    Decorator that can be used to mark functions as deprecated, emitting a warning when the function is used.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which emits a warning when used
    :type: function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapped function to be returned by the decorator.
        :param args: original function arguments
        :type: Tuple
        :param kwargs: original function keyword arguments
        :type: Dict
        :return: original function evaluation
        """
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn("Deprecated function {} invoked".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return wrapper


def kwargs_parameter_dict(func):
    """
    Decorator that passes all received `kwargs` as a keyword dictionary parameter.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which now has a new dictionary parameter called `parameter_dict` with all the
    original keyword arguments.
    :type: function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapped function to be returned by the decorator.
        :param args: original function arguments
        :type: Tuple
        :param kwargs: original function keyword arguments
        :type: Dict
        :return: original function evaluation
        """
        return func(*args, parameter_dict=kwargs, **kwargs)

    return wrapper


def log_time(func):
    """
    Decorator for logging the execution time of a function.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which now prints its execution time.
    :type: function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapped function to be returned by the decorator.
        :param args: original function arguments
        :type: Tuple
        :param kwargs: original function keyword arguments
        :type: Dict
        :return: original function evaluation
        """
        start = datetime.now()
        ret = func(*args, **kwargs)
        print("Executed {} in {} s.".format(func.__name__, datetime.now() - start))
        return ret

    return wrapper


def threaded(func):
    """
    Decorator that runs the decorated function asynchronously by throwing a new Python thread upon function call.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which evaluates in a new Python thread
    :type: function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapped function to be returned by the decorator.
        :param args: original function arguments
        :type: Tuple
        :param kwargs: original function keyword arguments
        :type: Dict
        :return: original function evaluation
        """
        t = Thread(target=func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()

    return wrapper


def torch_no_grad(func):
    """
    Decorator that runs the decorated function in a context with disabled PyTorch gradient calculation.
    Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
    It will reduce memory consumption for computations that would otherwise have requires_grad=True. In this mode, the
    result of every computation will have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    :param func: the function to be decorated
    :type: function
    :return: the decorated function, which evaluates in a context with disabled gradient calculations.
    :type: function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapped function to be returned by the decorator.
        :param args: original function arguments
        :type: Tuple
        :param kwargs: original function keyword arguments
        :type: Dict
        :return: original function evaluation
        """
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def parametrized(decorator):
    """
    Meta-decorator that adds parametrization support to other decorators.
    :param decorator: the decorator to be modified with parameter support
    :type: function
    :return: a decorator which can receive arguments and keyword arguments
    :type: function
    """

    @functools.wraps(decorator)
    def wrapper(*args, **kwargs):
        """
        Define and return the decorated decorator, which can receive arguments and keyword arguments.
        :param args: arguments to be received by the decorator
        :type: Tuple
        :param kwargs: keyword arguments to be received by the decorator
        :type: Dict
        :return: decorated function evaluation
        """

        def decorator_wrapper(func):
            """
            Evaluate the original decorator, which receives the function to be decorated along with the specified
            arguments and keyword arguments.
            :param func: the function to be decorated by the original decorator
            :type: function
            :return: the evaluation of the parametrized decorator
            """
            return decorator(func, *args, **kwargs)

        return decorator_wrapper

    return wrapper


@parametrized
def handle_exception_decorator(func, callback):
    """
    Method decorator that tries to execute the given function, and handles any exception by setting the
    object's status.
    :param func: the function to be decorated
    :type: Callable
    :param callback: function to execute on exception. Receives the instance and the error.
    :type: Callable
    :return: the decorated function
    :type: Callable
    """

    @functools.wraps(func)
    def wrapper(instance, *args, **kwargs):
        try:
            return func(instance, *args, **kwargs)
        except Exception as e:
            callback(instance, func, e, *args, **kwargs)

    return wrapper
