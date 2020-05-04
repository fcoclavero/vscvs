__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite trainer engines and common utilities. """


def attach_metrics(engine, metrics):
    """
    Attach all the metric objects in the `metrics` dictionary to the given Engine.
    :param engine: the ignite Engine to which the metrics should be attached.
    :type: ignite.Engine
    :param metrics: a dictionary with metric names as keys and metric objects as values.
    :type: ignite.metrics.Metric
    """
    for name, metric in metrics.items():
        metric.attach(engine, name)
