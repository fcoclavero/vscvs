__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for the Triplet Network architecture. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils.data import output_transform_triplet_evaluator as output_transform_evaluator, \
    output_transform_triplet_trainer as output_transform_trainer, \
    prepare_batch_multimodal as _prepare_batch


def create_triplet_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                           prepare_batch=_prepare_batch, output_transform=output_transform_trainer):
    """
    Factory function for creating an ignite trainer Engine for a triplet CNN.
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
    :param prepare_batch: batch preparation logic
    :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a triplet network trainer engine and returns value to
    be assigned to engine's state.output after each iteration.
    :type: Callable<args: `anchor_embeddings`, `positive_embeddings`, `negative_embeddings`, `loss`, ret: object>>
    (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device: model.to(device)

    def _update(_, batch):
        anchors, positives, negatives = prepare_batch(batch, device=device, non_blocking=non_blocking) # unpack batch
        optimizer.zero_grad() # reset gradients
        model.train() # training mode
        # Train over batch triplets - we assume batch items have their data in the `0` position
        anchor_embedding, positive_embedding, negative_embedding = model(anchors[0], positives[0], negatives[0])
        triplet_loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding) # compute the triplet loss
        triplet_loss.backward() # accumulate gradients
        optimizer.step() # update model wights
        return output_transform(anchor_embedding, positive_embedding, negative_embedding, triplet_loss)

    return Engine(_update)


def create_triplet_evaluator(model, metrics=None, device=None, non_blocking=False,
                             prepare_batch=_prepare_batch, output_transform=output_transform_evaluator):
    """
    Factory function for creating an evaluator for supervised models.
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
    :param prepare_batch: batch preparation logic
    :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a triplet network evaluator engine and returns the
    value to be assigned to engine's state.output after each iteration, which must fit that expected by the metrics.
    :type: Callable<args: `anchor_embeddings`, `positive_embeddings`, `negative_embeddings`,
                    ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device: model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            anchors, positives, negatives = prepare_batch(batch, device=device, non_blocking=non_blocking)
            anchor_embedding, positive_embedding, negative_embedding = model(anchors[0], positives[0], negatives[0])
            return output_transform(anchor_embedding, positive_embedding, negative_embedding)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine
