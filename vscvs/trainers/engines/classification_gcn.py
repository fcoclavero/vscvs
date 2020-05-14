__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for a classification GCN. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils import output_transform_evaluator, output_transform_trainer, prepare_batch_graph as _prepare_batch


def create_classification_gcn_trainer(model, optimizer, loss_fn, classes_dataframe, device=None, non_blocking=False,
                                      processes=None, prepare_batch=_prepare_batch,
                                      output_transform=output_transform_trainer):
    """
    Factory function for creating an ignite trainer Engine for a classification GCN.
    :param model: the generator model - generates vectors from images.
    :type: torch.nn.Module
    :param optimizer: the optimizer to be used for the generator model.
    :type: torch.optim.Optimizer
    :param loss_fn: the triplet loss
    :type: torch.nn.Module
    :param classes_dataframe: dataframe containing class names and their word vectors.
    :type: pandas.Dataframe
    :param device: (optional) (default: None) device type specification.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int
    :param prepare_batch: batch preparation logic that takes a simple `x` and `y` batch and returns the
    corresponding batch graph.
    :type: Callable[[List[torch.Tensor], pandas.Dataframe], Tuple[torch_geometric.data.Data, torch.Tensor]]
    :param output_transform: (optional) function that receives the result of a typical network trainer engine (the
    input elements, labels, network outputs and the loss module) and returns value to be assigned to the engine's
    `state.output` after each iteration, typically the loss value.
    :type: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module], float]
    :return: a trainer engine with the update function.
    :type: ignite.engine.Engine
    """
    if device: model.to(device)

    def _update(_, batch):
        model.train() # # set training mode
        optimizer.zero_grad() # reset gradients
        batch_graph = prepare_batch(
            batch, classes_dataframe, device=device, non_blocking=non_blocking, processes=processes)
        y_pred = model(batch_graph) # feed data to model
        loss = loss_fn(y_pred, batch_graph.y) # compute loss
        loss.backward() # back propagation
        optimizer.step() # update model wights
        return output_transform(*batch, y_pred, loss)

    return Engine(_update)


def create_classification_gcn_evaluator(model, classes_dataframe, metrics=None, device=None, non_blocking=False,
                                        processes=None, prepare_batch=_prepare_batch,
                                        output_transform=output_transform_evaluator):
    """
    Factory function for creating an evaluator for a classification GCN.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param model: the model to train.
    :type: torch.nn.Module
    :param classes_dataframe: dataframe containing class names and their word vectors.
    :type: pandas.Dataframe
    :param metrics: map of metric names to Metrics.
    :type: Dict[str, ignite.metrics.Metric]]
    :param device: (optional) (default: None) device type specification. Applies to both model and batches.
    :type: str
    :param non_blocking: (optional) if True and the copy is between CPU and GPU, the copy may run asynchronously.
    :type: bool
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int
    :param prepare_batch: batch preparation logic that takes a simple `x` and `y` batch and returns the
    corresponding batch graph.
    :type: Callable[[List[torch.Tensor], pandas.Dataframe], Tuple[torch_geometric.data.Data, torch.Tensor]]
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
            batch_graph = prepare_batch(
                batch, classes_dataframe, device=device, non_blocking=non_blocking, processes=processes)
            x, y = batch_graph.x, batch_graph.y
            y_pred = model(batch_graph) # feed data to model
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine
