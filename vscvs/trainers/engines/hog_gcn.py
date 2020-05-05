__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for a GCN image label classifier, using HOG feature vectors for images. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils import output_transform_evaluator, output_transform_trainer, prepare_batch as _prepare_batch


def create_hog_gcn_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                           prepare_batch=_prepare_batch, output_transform=output_transform_trainer):
    """
    Factory function for creating an ignite trainer Engine for a class classification GCN that creates batch clique
    graphs where node feature vectors correspond to batch image HOG feature vectors and vertex weights correspond to the
    distance of class name strings' document vectors.
    :param model: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param loss_fn: the triplet loss
    :type: torch.nn loss function
    :param device: device type specification
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: image batch preparation logic
    :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch.Tensor, torch.Tensor>>
    :param output_transform: function that receives the result of the network trainer engine and returns value to
    be assigned to engine's state.output after each iteration.
    :type: Callable<args: `x`, `y`, `y_pred`, `loss`, ret: object>> (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device: model.to(device)

    def _update(_, batch):
        model.train() # # set training mode
        optimizer.zero_grad() # reset gradients
        image_batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(image_batch) # feed data to model
        loss = loss_fn(y_pred, image_batch[1]) # compute loss
        loss.backward() # back propagation
        optimizer.step() # update model wights
        return output_transform(*image_batch, y_pred, loss)

    return Engine(_update)


def create_hog_gcn_evaluator(model, metrics=None, device=None, non_blocking=False,
                             prepare_batch=_prepare_batch, output_transform=output_transform_evaluator):
    """
    Factory function for creating an evaluator for a class classification GCN that creates batch clique graphs where
    node feature vectors correspond to batch image HOG feature vectors and vertex weights correspond to the distance of
    class name strings' document vectors.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param model: the model to train.
    :type: torch.nn.Module
    :param metrics: map of metric names to Metrics.
    :type: dict<str:<ignite.metrics.Metric>>
    :param device: device type specification. Applies to both model and batches.
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: image batch preparation logic
    :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch_geometric.data.Data, torch.Tensor>>
    :param output_transform: function that receives the result of the network trainer engine and returns value to
    be assigned to engine's state.output after each iteration, which myst fit that expected by the metrics.
    :type: Callable<args: `x`, `y`, `y_pred` , ret: tuple<torch.Tensor, torch.Tensor>> (optional)
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device: model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            image_batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            x, y, *_ = image_batch
            y_pred = model(image_batch)  # feed data to model
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine
