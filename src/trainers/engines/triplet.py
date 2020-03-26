__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for the Triplet Network architecture. """


import torch

import torch.nn.functional as F

from ignite.engine import Engine

from src.utils.data import output_transform_triplet, prepare_batch_triplet


def create_triplet_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                           prepare_batch=prepare_batch_triplet):
    """
    Factory function for creating an ignite trainer Engine for a triplet CNN.
    :param model: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param loss_fn: the triplet loss
    :type: torch.nn loss function
    :param vector_dimension: the dimensionality of the common vector space.
    :type: int
    :param device: device type specification
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable (args:`batch`, `device`, `non_blocking`, ret:tuple<torch.Tensor, torch.Tensor> (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        # Unpack batch
        anchors, positives, negatives = prepare_batch(batch, device=device, non_blocking=non_blocking)
        # Reset gradients
        optimizer.zero_grad()
        # Training mode
        model.train()
        # Train over batch triplets - we assume batch items have their data in the `0` position
        anchor_embedding, positive_embedding, negative_embedding = model(anchors[0], positives[0], negatives[0])
        distance_to_positive = F.pairwise_distance(anchor_embedding, positive_embedding, 2)
        distance_to_negative = F.pairwise_distance(anchor_embedding, negative_embedding, 2)
        # Create target tensor. A target of -1 denotes that the first input should be ranked lower (have lesser value)
        # than the second input, fitting our case: first (second) input is distance to positive (negative). See (b)
        target = -torch.ones(anchors[1].size()[0], device=device) # anchors[1] are the classes, shape = `[batch_size]`
        # (b) Compute the triplet loss: https://pytorch.org/docs/stable/nn.html#torch.nn.MarginRankingLoss
        triplet_loss = loss_fn(distance_to_positive, distance_to_negative, target) # = d2positive - d2negative + margin
        # Accumulate gradients
        triplet_loss.backward()
        # Update model wights
        optimizer.step()

        def accuracy(dista, distb):
            margin_acc = 0
            pred = (dista - distb - margin_acc).cpu().data
            return (pred > 0).sum() * 1.0 / dista.size()[0]

        # Return loss and average distances for logging
        return triplet_loss, torch.mean(distance_to_positive), torch.mean(distance_to_negative), \
               accuracy(distance_to_positive, distance_to_negative)

    return Engine(_update)


def create_triplet_evaluator(model, metrics={}, device=None, non_blocking=False, prepare_batch=prepare_batch_triplet,
                             output_transform=output_transform_triplet):
    """
    Factory function for creating an evaluator for supervised models.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param model: the model to train.
    :type: torch.nn.Module
    :param: metrics: map of metric names to Metrics.
    :type: dict<str:<ignite.metrics.Metric>>
    :param device: device type specification. Applies to both model and batches.
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable (args:`batch`, `device`, `non_blocking`, ret:tuple<torch.Tensor, torch.Tensor> (optional)
    :param output_transform: function that receives `x`, `y`, `y_pred` and the returns value to be assigned to engine's
    state.output after each iteration. Default is returning `(y_pred, y,)`, which fits output expected by metrics.
    If you change it you should use `output_transform` in metrics.
    :type: Callable (args:`x`, `y`, `y_pred`, ret:tuple<torch.Tensor, torch.Tensor> (optional)
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine