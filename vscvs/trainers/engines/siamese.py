__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Ignite trainer engine (training logic) for a siamese architecture. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils import output_transform_siamese_evaluator as output_transform_evaluator
from vscvs.utils import output_transform_siamese_trainer as output_transform_trainer
from vscvs.utils import prepare_batch_siamese as _prepare_batch


def create_siamese_trainer(
    model,
    optimizer,
    loss_fn,
    device=None,
    non_blocking=False,
    prepare_batch=_prepare_batch,
    output_transform=output_transform_trainer,
):
    """
    Factory function for creating an ignite trainer Engine for a siamese architecture.
    :param model: siamese network module that receives, embeds and computes embedding distances between image pairs.
    :type: SiameseNetwork
    :param optimizer: the optimizer to be used for the siamese network
    :type: torch.optim.Optimizer
    :param loss_fn: contrastive loss function
    :type: torch.nn.Module
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool
    :param prepare_batch: (optional) batch preparation logic. Takes a batch, the device and the `non_blocking`
    option and returns the triplet tensors: the siamese pair tensors and the siamese target indicating pair similarity.
    :type: Callable[[Tuple[List[torch.Tensor], List[torch.Tensor]]], str, bool],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :param output_transform: (optional) function that receives the result of a siamese network trainer engine (the
    siamese pair embeddings, siamese target and the loss module) and returns value to be assigned to the engine's
    `state.output` after each iteration, typically the loss value.
    :type: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module], float]
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _update(_, batch):
        images_1, images_2, target = prepare_batch(batch, device=device, non_blocking=non_blocking)  # unpack batch
        optimizer.zero_grad()  # reset gradients
        model.train()  # training mode
        embeddings_0, embeddings_1 = model(images_1[0], images_2[0])  # train over batch pairs. Actual images in `0` idx
        contrastive_loss = loss_fn(embeddings_0, embeddings_1, target)  # compute the contrastive loss
        contrastive_loss.backward()  # accumulate gradients
        optimizer.step()  # update model weights
        return output_transform(embeddings_0, embeddings_1, target, contrastive_loss)

    return Engine(_update)


def create_siamese_evaluator(
    model,
    metrics=None,
    device=None,
    non_blocking=False,
    prepare_batch=_prepare_batch,
    output_transform=output_transform_evaluator,
):
    """
    Factory function for creating an evaluator for supervised models.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param model: the model to train.
    :type: torch.nn.Module
    :param metrics: map of metric names to Metrics.
    :type: Dict[str, ignite.metrics.Metric]]
    :param device: (optional) (default: None) device type specification. Applies to both model and batches.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param prepare_batch: (optional) batch preparation logic. Takes a batch, the device and the `non_blocking`
    option and returns the triplet tensors: the siamese pair tensors and the siamese target indicating pair similarity.
    :type: Callable[[Tuple[List[torch.Tensor], List[torch.Tensor]]], str, bool],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :param output_transform: (optional) function that receives the result of a siamese network evaluator engine (the
    siamese pair embeddings and the siamese target) and returns the value to be assigned to the engine's `state.output`
    after each iteration, which must fit that expected by the metrics, typically all three input tensors.
    :type: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _inference(_, batch):
        model.eval()
        with torch.no_grad():
            images_0, images_1, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
            embeddings_0, embeddings_1 = model(images_0[0], images_1[0])
            return output_transform(embeddings_0, embeddings_1, target)

    engine = Engine(_inference)
    if metrics:
        attach_metrics(engine, metrics)
    return engine
