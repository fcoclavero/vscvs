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
    :type: torch.nn.Module
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param prepare_batch: (optional) batch preparation logic. Takes a batch, the device and the `non_blocking`
    option and returns the batch elements and labels.
    :type: Callable[[List[torch.Tensor]], str, bool], Tuple[torch.Tensor, torch.Tensor]]
    :param output_transform: (optional) function that receives the result of a typical network trainer engine (the
    input elements, labels, network outputs and the loss module) and returns value to be assigned to the engine's
    `state.output` after each iteration, typically the loss value.
    :type: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module], float]
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
    :type: dict[str, ignite.metrics.Metric]
    :param device: (optional) (default: None) device type specification. Applies to both model and batches.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param prepare_batch: (optional) batch preparation logic. Takes a batch, the device and the `non_blocking`
    option and returns the batch elements and labels.
    :type: Callable[[List[torch.Tensor]], str, bool], Tuple[torch.Tensor, torch.Tensor]]
    :param output_transform: (optional) function that receives the result of the network evaluator engine (the input
    elements, labels and network outputs) and returns value to be assigned to the engine's `state.output` after each
    iteration, which must fit that expected by the metrics, typically the network output followed by the labels.
    :type: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
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
