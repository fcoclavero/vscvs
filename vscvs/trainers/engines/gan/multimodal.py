__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for multimodal GAN architecture. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils.data import output_transform_multimodal_gan_evaluator as output_transform_evaluator, \
    output_transform_multimodal_gan_trainer as output_transform_trainer, \
    prepare_batch_multimodal as _prepare_batch


def create_multimodal_gan_trainer(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn,
                                  device=None, non_blocking=False, prepare_batch=_prepare_batch,
                                  output_transform=output_transform_trainer):
    """
    Factory function for creating an ignite trainer Engine for a multimodal GAN.
    NOTES:
        * The model can be trained to create multimodal spaces for any number of modes.
        * Batches contain `batch_size` elements, each of which is represented `n_mode` times, in each of the modes to be
          included in the resulting common vector space.
        * The generator ($G$) is composed of `n_modes` networks, each of which performs the transformation of elements
          from each mode space to the common vector space (embeddings).
        * The discriminator ($D$) is trained to classify generator embeddings into each of the corresponding modes. It
          must try to produce a probability of one for the original mode and a probability of zero for the others. One-
          hot-encodings are used to mark the original mode, thus the discriminator output must try to match these.
        * The generator is optimized to maximize the mistakes the discriminator makes. The same loss function can be
          simply changing the target vector to have zero probability in the correct mode and an evenly distributed
          probability in the rest of the modes. For example `[0, 1, 0]` -> `[0.5, 0, 0.5]`.
        * Given the generator formulation, a multi-class classification or a regression loss function must be used.
    :param generator: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
    :type: torch.nn.Module
    :param generator_optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param discriminator_optimizer: the optimizer to be used for the discriminator model
    :type: torch.optim.Optimizer
    :param loss_fn: the loss function for the GAN model
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
    if device:
        generator.to(device)
        discriminator.to(device)

    def _update(_, batch):

        ############################
        # (0) Prepare batch and labels, and make a forward pass through the models.
        ###########################

        batch =  prepare_batch(batch, device=device, non_blocking=non_blocking)
        n_modes, batch_size, classes, mode_labels, generator_labels = _multimodal_batch_variables(batch, device)

        ############################
        # (1) Update G network: maximize log(D(G(z)))
        ###########################

        generator.zero_grad()
        embedding_list = generator(*[sub_batch[0] for sub_batch in batch])  # forward pass with same mode sub-batches
        embeddings = torch.cat(embedding_list)  # create a single discriminator batch from sub-batch list
        generator_loss = loss_fn(discriminator(embeddings), generator_labels)
        generator_loss.backward()
        generator_optimizer.step()

        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        discriminator.zero_grad()
        mode_predictions = discriminator(embeddings.detach())
        discriminator_loss = loss_fn(mode_predictions, mode_labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        return output_transform(
            embeddings, mode_predictions, mode_labels, generator_labels, classes, generator_loss, discriminator_loss)

    return Engine(_update)


def create_multimodal_gan_evaluator(generator, discriminator, metrics=None, device=None, non_blocking=False,
                                    prepare_batch=_prepare_batch, output_transform=output_transform_evaluator):
    """
    Factory function for creating an evaluator for GAN models.
    NOTE: `engine.state.output` for this engine is defined by `output_transform` parameter and is
    a tuple of `(batch_pred, batch_y)` by default.
    :param generator: the generator model.
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
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
    if device:
        generator.to(device)
        discriminator.to(device)

    def _inference(_, batch):
        generator.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            n_modes, batch_size, classes, mode_labels, generator_labels = _multimodal_batch_variables(batch, device)
            embeddings, mode_predictions = _forward_pass(generator, discriminator, batch)
            return output_transform(embeddings, mode_predictions, mode_labels, generator_labels, classes)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine


def _multimodal_batch_variables(batch, device):
    """
    Compute batch-derived variables needed for multimodal GAN processing:
    - `n_modes`: the number of different modalities present in the batch.
    - `batch_size`: the number of batch elements.
    - `classes`: the classes of the entities present in the batch.
    - `mode_labels`: labels indicating the mode of each element if sub-batch embeddings are stacked. One-hot encoded.
    - `generator_labels`: the target vector for using the same loss function for the generator loss. In this case, ins
    :param batch: tuple with the multimodal batch to be fed into the generator network.
    :type: tuple<torch.Tensor, ...>
    :param device: the device type specification where the processing is to take place.
    :type: str of torch.device (optional) (default: None)
    :return: the batch-derived variables: `n_modes`, `batch_size`, `classes`, `mode_labels`, `generator_labels`.
    :type: tuple
    """
    n_modes = len(batch)
    batch_size = len(batch[0][0])  # any mode should have same lengths
    classes = batch[0][1]  # any mode should have the same class idx
    mode_labels = torch.cat([t.expand(batch_size, n_modes) for t in torch.diag(torch.ones(n_modes))]).to(device)
    # noinspection PyTypeChecker
    generator_labels = (1 - mode_labels.float()) / (n_modes - 1) # type is automatically inferred
    return n_modes, batch_size, classes, mode_labels, generator_labels


def _forward_pass(generator, discriminator, batch):
    """
    Perform a forward pass through the multimodal GAN, generating embeddings for batch elements and performing mode
    predictions for each of the resulting embeddings.
    :param generator: the generator model.
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
    :type: torch.nn.Module
    :param batch: tuple with the multimodal batch to be fed into the generator network.
    :type: tuple<torch.Tensor, ...>
    :return: the generator embeddings and the discriminator mode predictions.
    :type: tuple
    """
    embedding_list = generator(*[sub_batch[0] for sub_batch in batch])  # forward pass with same mode sub-batches
    embeddings = torch.cat(embedding_list)  # create a single discriminator batch from sub-batch list
    mode_predictions = discriminator(embeddings)
    return embeddings, mode_predictions