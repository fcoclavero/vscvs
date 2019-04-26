import torch

from ignite.engine import Engine

from src.utils.data import prepare_batch


def create_triplet_cnn_trainer(model, optimizer, loss, vector_dimension, device=None, non_blocking=False,
                               prepare_batch=prepare_batch):
    """
    Factory function for creating an ignite trainer Engine for a triplet CNN.
    :param model: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param loss: the loss function for the GAN model
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
        photos, sketches, classes = batch # TODO: change to accomodate triplets
        # Reset gradients
        model.zero_grad()
        # Training mode
        model.train()

        ############################
        # (1) Anchor
        ###########################
        photos_vectors = model(photos).view(-1, vector_dimension)

        ############################
        # (2) Positive
        ###########################

        ############################
        # (3) Negative
        ###########################

        ############################
        # (4) Update network
        ###########################
        # Calculate the discriminator loss
        photo_discriminator_loss = loss(photos_class_prediction, photo_labels)
        # Accumulate gradients
        photo_discriminator_loss.backward()
        # Update model wights
        optimizer.step()
        # Return losses, for logging
        return network_loss

    return Engine(_update)