__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for a siamese architecture. """


import torch

from ignite.engine import Engine

from src.utils.data import prepare_batch_siamese, output_transform_siamese


def create_siamese_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                           prepare_batch=prepare_batch_siamese):
    """
    Factory function for creating an ignite trainer Engine for a siamese architecture.
    :param model: siamese network module that receives, embeds and computes embedding distances between image pairs.
    :type: SiameseNetwork
    :param optimizer: the optimizer to be used for the siamese network
    :type: torch.optim.Optimizer
    :param loss_fn: contrastive loss function
    :type: torch.nn loss function
    :param device: device type specification
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable<args:`batch`,`device`,`non_blocking`, ret:tuple<torch.Tensor,torch.Tensor,torch.Tensor>> (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        # Unpack batch
        images_1, images_2, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
        # Reset gradients
        optimizer.zero_grad()
        # Training mode
        model.train()
        # Train over batch triplets - we assume batch items have their data in the `0` position
        embeddings_0, embeddings_1 = model(images_1[0], images_2[0])
        # Compute the contrastive loss
        contrastive_loss = loss_fn(embeddings_0, embeddings_1, target)
        # Accumulate gradients
        contrastive_loss.backward()
        # Update model wights
        optimizer.step()
        # Return loss for logging
        return contrastive_loss.item()

    return Engine(_update)


def create_siamese_evaluator(model, metrics=None, device=None, non_blocking=False,
                             prepare_batch=prepare_batch_siamese, output_transform=output_transform_siamese):
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
    :param output_transform: function that receives `x`, `y`, `y_pred` and the returns value to be assigned to engine's
    state.output after each iteration. Default is returning `(y_pred, y,)`, which fits output expected by metrics.
    If you change it you should use `output_transform` in metrics.
    :type: Callable<args: `x`, `y`, `y_pred`, ret: tuple<torch.Tensor, torch.Tensor>> (optional)
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            images_0, images_1, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
            embeddings_0, embeddings_1 = model(images_0[0], images_1[0])
            return output_transform(embeddings_0, embeddings_1, target)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine