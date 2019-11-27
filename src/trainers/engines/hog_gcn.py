__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for a GCN image label classifier, using HOG feature vectors for images. """


import torch

from ignite.engine import Engine


def create_hog_gcn_trainer(prepare_batch, model, classes_dataframe, optimizer, loss_fn, device=None,
                           non_blocking=False, processes=None):
    """
    Factory function for creating an ignite trainer Engine for a class classification GCN that creates batch clique
    graphs where node feature vectors correspond to batch image HOG feature vectors and vertex weights correspond to the
    distance of class name strings' document vectors.
    :param prepare_batch: image batch preparation logic
    :type: Callable (args:`batch`,`device`,`non_blocking`, ret:tuple(torch.Tensor, torch.Tensor)
    :param model: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param classes_dataframe: dataframe containing class names and their word vectors
    :type: pandas.Dataframe
    :param optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param loss_fn: the triplet loss
    :type: torch.nn loss function
    :param device: device type specification
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int or None
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train() # # set training mode
        optimizer.zero_grad() # reset gradients
        image_batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(image_batch) # feed data to model
        loss = loss_fn(y_pred, image_batch[1]) # compute loss
        loss.backward() # back propagation
        optimizer.step() # update model wights
        return loss.item() # return loss for logging

    return Engine(_update)


def create_hog_gcn_evaluator(prepare_batch, model, classes_dataframe, metrics=None, device=None,
                             non_blocking=False, processes=None):
    """
    Factory function for creating an evaluator for a class classification GCN that creates batch clique graphs where
    node feature vectors correspond to batch image HOG feature vectors and vertex weights correspond to the distance of
    class name strings' document vectors.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param prepare_batch: image batch preparation logic
    :type: Callable (args:`batch`,`device`,`non_blocking`, ret:tuple(torch_geometric.data.Data, torch.Tensor)
    :param model: the model to train.
    :type: torch.nn.Module
    :param classes_dataframe: dataframe containing class names and their word vectors
    :type: pandas.Dataframe
    :param metrics: map of metric names to Metrics.
    :type: dict<str:<ignite.metrics.Metric>>
    :param device: device type specification. Applies to both model and batches.
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int or None
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _output_transform(x, y, y_pred):
        """ Value to be assigned to engine's state.output after each iteration. """
        return y_pred, y  # output format is according to `Accuracy` docs

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            image_batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            x, y, *_ = image_batch
            y_pred = model(image_batch)  # feed data to model
            return _output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
