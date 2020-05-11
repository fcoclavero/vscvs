__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Engine output transforms. """


def output_transform_evaluator(_x, y, y_pred):
    """
    Value to be assigned to engine's `state.output` after each iteration. This is the default format for classifiers.
    :param _x: the input tensor.
    :type: torch.Tensor
    :param y: the label or target tensor.
    :param y_pred: the output of the model.
    :type: torch.Tensor
    :return: the expected values for a default Ignite Metric, `y_pred` and `y`.
    type: tuple<torch.Tensor. torch.Tensor>
    """
    return y_pred, y


def output_transform_trainer(_x, _y, _y_pred, loss):
    """
    Value to be assigned to engine's `state.output` after each iteration. This is the default format for classifiers.
    :param _x: the input tensor.
    :type: torch.Tensor
    :param _y: the label or target tensor.
    :param _y_pred: the output of the model.
    :type: torch.Tensor
    :param loss: the loss module for the network.
    :type: torch.nn.Module
    :return: value to be assigned to engine's `state.output` after each iteration, which by default is the loss value.
    :type: tuple<torch.Tensor>
    """
    return loss.item()


def output_transform_multimodal_gan_evaluator(embeddings, mode_predictions, mode_labels, generator_labels, classes):
    """
    Receives the result of a multimodal GAN evaluator engine and returns value to be assigned to engine's `state.output`
    after each iteration.
    :param embeddings: tensor with the embedding vectors for each batch element (thus one embedding per mode per
    entity). The tensor has a shape of `[n_modes * batch_size, embedding_length]`.
    :type: list<torch.Tensor>
    :param mode_predictions: tensor of shape `[n_modes * batch_size, n_modes]` with the probability of each element
    to belong to each mode.
    :type: torch.Tensor
    :param mode_labels: the actual mode labels for each element. Tensor of shape `[n_modes * batch_size, n_modes]`.
    :type: torch.Tensor
    :param generator_labels: the labels used for the generator loss, usually the additive inverse of each mode label.
    :type: torch.Tensor
    :param classes: the classes, or categories, of each entity in the dataset. Tensor of shape `[batch_size]`.
    :type: torch.Tensor
    :return: the value to be assigned to engine's `state.output` after each iteration, which must fit that expected by the
    metrics, which by default in a multimodal GAN is the element embeddings, mode predictions and labels, and classes.
    :type: tuple<list<torch.Tensor>, torch.Tensor, torch.Tensor, torch.Tensor>
    """
    return embeddings, mode_predictions, mode_labels, generator_labels, classes


def output_transform_multimodal_gan_trainer(_embeddings, _mode_predictions, _mode_labels, _generator_labels, _classes,
                                            generator_loss, discriminator_loss):
    """
    Receives the result of a multimodal GAN trainer engine and returns value to be assigned to engine's `state.output`
    after each iteration, which by default is the loss values.
    :param _embeddings: list with the embedding vectors for each batch element of each mode (thus one embedding per mode
    per entity). The list has a length equal to the number of modes and each embedding tensor has a shape of
    `[n_modes, embedding_length]`.
    :type: list<torch.Tensor>
    :param _mode_predictions: tensor of shape `[n_modes * batch_size, n_modes]` with the probability of each element
    to belong to each mode.
    :type: torch.Tensor
    :param _mode_labels: the actual mode labels for each element. Tensor of shape `[n_modes * batch_size, n_modes]`.
    :type: torch.Tensor
    :param _generator_labels: the labels used for the generator loss, usually the additive inverse of each mode label.
    :type: torch.Tensor
    :param _classes: the classes, or categories, of each entity in the dataset. Tensor of shape `[batch_size]`.
    :type: torch.Tensor
    :param generator_loss: the loss module for the generator network.
    :type: torch.nn.Module
    :param discriminator_loss: the loss module for the discriminator network.
    :type: torch.nn.Module
    :return: value to be assigned to engine's `state.output` after each iteration.
    :type: tuple<torch.Tensor, torch.Tensor>
    """
    return generator_loss.item(), discriminator_loss.item()


def output_transform_multimodal_gan_siamese_trainer(
        _embeddings_0, _embeddings_1, _siamese_target, _mode_predictions, _mode_labels, _generator_labels,
        generator_loss, discriminator_loss):
    """
    Receives the result of a multimodal GAN siamese trainer engine and returns value to be assigned to engine's
    `state.output` after each iteration, which by default is the loss values.
    :param _embeddings_0: torch tensor containing the embeddings for the first image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _embeddings_1: torch tensor containing the embeddings for the second image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _siamese_target: tensor with the contrastive loss target for each pair (0 for similar images, 1 otherwise).
    :type: torch.Tensor
    :param _mode_predictions: tensor of shape `[n_modes * batch_size, n_modes]` with the probability of each element
    to belong to each mode.
    :type: torch.Tensor
    :param _mode_labels: the actual mode labels for each element. Tensor of shape `[n_modes * batch_size, n_modes]`.
    :type: torch.Tensor
    :param _generator_labels: the labels used for the generator loss, usually the additive inverse of each mode label.
    :type: torch.Tensor
    :param generator_loss: the loss module for the generator network.
    :type: torch.nn.Module
    :param discriminator_loss: the loss module for the discriminator network.
    :type: torch.nn.Module
    :return: value to be assigned to engine's `state.output` after each iteration.
    :type: tuple<torch.Tensor, torch.Tensor>
    """
    return generator_loss.item(), discriminator_loss.item()


def output_transform_siamese_evaluator(embeddings_0, embeddings_1, target):
    """
    Receives the result of a siamese network evaluator engine (the embeddings of each image and the target tensor) and
    returns value to be assigned to engine's `state.output` after each iteration.
    :param embeddings_0: torch tensor containing the embeddings for the first image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param embeddings_1: torch tensor containing the embeddings for the second image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param target: tensor with the contrastive loss target for each pair (0 for similar images, 1 otherwise).
    :type: torch.Tensor
    :return: the value to be assigned to engine's `state.output` after each iteration, which must fit that expected by the
    metrics. By default, in a siamese network, it is the embeddings of each image pair and their target tensor.
    :type: tuple<torch.Tensor>
    """
    return embeddings_0, embeddings_1, target


def output_transform_siamese_trainer(_embeddings_0, _embeddings_1, _target, loss):
    """
    Receives the result of a siamese network trainer engine (the embeddings of each image, the target tensor and the
    loss module) and returns value to be assigned to engine's `state.output` after each iteration.
    :param _embeddings_0: torch tensor containing the embeddings for the first image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _embeddings_1: torch tensor containing the embeddings for the second image of each image pair.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _target: tensor with the contrastive loss target for each pair (0 for similar images, 1 otherwise).
    :type: torch.Tensor
    :param loss: the loss module.
    :type: torch.nn.Module
    :return: value to be assigned to engine's `state.output` after each iteration, which by default is the loss value.
    :type: tuple<torch.Tensor>
    """
    return loss.item()


def output_transform_triplet_evaluator(anchor_embeddings, positive_embeddings, negative_embeddings):
    """
    Receives the result of a triplet network evaluator engine (the embeddings of each triplet element) and
    returns value to be assigned to engine's `state.output` after each iteration.
    :param anchor_embeddings: torch tensor containing the embeddings for the anchor elements.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param positive_embeddings: torch tensor containing the embeddings for the positive elements (anchor class).
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param negative_embeddings: torch tensor containing the embeddings for the negative elements (different class).
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :return: value to be assigned to engine's `state.output` after each iteration, which must fit that expected by the
    metrics. By default, in a triplet network, it is the embeddings of each triplet.
    :type: tuple<torch.Tensor>
    """
    return anchor_embeddings, positive_embeddings, negative_embeddings


def output_transform_triplet_trainer(_anchor_embeddings, _positive_embeddings, _negative_embeddings, loss):
    """
    Receives the result of a triplet network trainer engine (the embeddings of each triplet element and the loss
    module) and returns value to be assigned to engine's `state.output` after each iteration.
    :param _anchor_embeddings: torch tensor containing the embeddings for the anchor elements.
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _positive_embeddings: torch tensor containing the embeddings for the positive elements (anchor class).
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param _negative_embeddings: torch tensor containing the embeddings for the negative elements (different class).
    :type: torch.Tensor with shape `(embedding_size, batch_size)`
    :param loss: the loss module.
    :type: torch.nn.Module
    :return: value to be assigned to engine's `state.output` after each iteration, which by default is the loss value.
    :type: tuple<torch.Tensor>
    """
    return loss.item()
