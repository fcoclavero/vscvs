import torch

from ignite.engine import Engine

from src.utils.data import prepare_batch


def create_triplet_cnn_trainer(model, optimizer, loss_fn, vector_dimension, device=None, non_blocking=False,
                               prepare_batch=prepare_batch):
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
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable (args:`batch`,`device`,`non_blocking`, ret:tuple(torch.Tensor,torch.Tensor) (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    print(model)

    if device:
        model.to(device)

    def _update(engine, batch):
        # Unpack batch
        anchors, positives, negatives = batch
        # Reset gradients
        optimizer.zero_grad()
        # Training mode
        model.train()
        # Train over batch triplets - we assume batch items have their data in the `0` position
        anchor_embedding, positive_embedding, negative_embedding, \
            distance_to_positive, distance_to_negative = model(anchors[0], positives[0], negatives[0])
        # Create target tensor. A target of 1 denotes that the first input should be ranked higher (have a larger value)
        # than the second input, fitting our case: first (second) input is distance to positive (negative). See (b)
        target = torch.FloatTensor(anchors[1].size()[0]) # anchors[1] are the classes, Tensor of shape = `[batch_size]`
        # Compute the triplet loss
        triplet_loss = loss_fn(distance_to_positive, distance_to_negative, target) # (b)
        # Embedding loss
        # embedding_loss = anchor_embedding.norm(2) + positive_embedding.norm(2) + negative_embedding.norm(2)
        # loss = triplet_loss + 0.001 * embedding_loss
        # Accumulate gradients
        triplet_loss.backward()
        # loss.backward()
        # Update model wights
        optimizer.step()
        # Return loss for logging
        return triplet_loss

    return Engine(_update)