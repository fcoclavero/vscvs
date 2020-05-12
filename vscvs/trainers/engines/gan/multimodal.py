__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer engine (training logic) for multimodal GAN architecture. """


import torch

from ignite.engine import Engine

from vscvs.trainers.engines import attach_metrics
from vscvs.utils import output_transform_multimodal_gan_evaluator as output_transform_evaluator, \
    output_transform_multimodal_gan_siamese_evaluator as output_transform_evaluator_siamese, \
    output_transform_multimodal_gan_trainer as output_transform_trainer, \
    output_transform_multimodal_gan_siamese_trainer as output_transform_trainer_siamese, \
    prepare_batch_multimodal as _prepare_batch, \
    prepare_batch_multimodal_siamese as prepare_batch_siamese


def combine_batches(*batches):
    """
    Combine two multimodal batches.
    :param batches: list of the multimodal batch tuples to be combined.
    :type: list<tuple<torch.Tensor, ...>>
    :return: a combined multimodal batch, with a `batch_size` that is equal to the sum of the batch sizes of the batches
    in `batches`.
    :type: tuple<torch.Tensor, ...>
    """
    return tuple((torch.cat([batch[i][0] for batch in batches]), torch.cat([batch[i][1] for batch in batches]))
                 for i in range(len(batches[0])))


def prepare_multimodal_batch_variables(batch, device):
    """
    Compute batch-derived variables needed for multimodal GAN processing:
    - `classes`: the classes of the entities present in the batch.
    - `mode_labels`: labels indicating the mode of each element if sub-batch embeddings are stacked. One-hot encoded.
    - `generator_labels`: the target vector for using the same loss function for the generator loss. In this case a
      tensor with zero in the correct mode and an evenly distributed probability in the rest of the modes.
      For example `[0, 1, 0]` -> `[0.5, 0, 0.5]`.
    :param batch: tuple with the multimodal batch to be fed into the generator network.
    :type: tuple<torch.Tensor, ...>
    :param device: the device type specification where the processing is to take place.
    :type: str of torch.device (optional) (default: None)
    :return: the batch-derived variables: `batch_size`, `classes`, `mode_labels`, `generator_labels`.
    :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
    """
    n_modes = len(batch)
    batch_size = len(batch[0][0])  # any mode should have same lengths
    classes = batch[0][1]  # any mode should have the same class idx
    mode_labels = torch.cat([t.expand(batch_size, n_modes) for t in torch.diag(torch.ones(n_modes))]).to(device)
    # noinspection PyTypeChecker
    generator_labels = (1 - mode_labels.float()) / (n_modes - 1) # type is automatically inferred
    return classes, mode_labels, generator_labels


def prepare_bimodal_batch_variables(batch, device):
    """
    Alternative to `prepare_multimodal_batch_variables` that can be used in a bimodal setting, enabling the use of
    the BCELoss instead of a multi-class classification or regression loss function. It modifies the following:
    - `mode_labels`: labels indicating the mode of each element, which in this scenario are binary.
    - `generator_labels`: the target vector for using the same loss function for the generator loss. In this case it
      corresponds to `1 - mode_labels`.
    :param batch: tuple with the multimodal batch to be fed into the generator network.
    :type: tuple<torch.Tensor, ...>
    :param device: the device type specification where the processing is to take place.
    :type: str of torch.device (optional) (default: None)
    :return: the batch-derived variables: `n_modes`, `batch_size`, `classes`, `mode_labels`, `generator_labels`.
    :type: tuple
    """
    batch_size = len(batch[0][0])  # any mode should have same lengths
    classes = batch[0][1]  # any mode should have the same class idx
    mode_labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).to(device)
    # noinspection PyTypeChecker
    generator_labels = (1 - mode_labels) # type is automatically inferred
    return classes, mode_labels, generator_labels


def prepare_bimodal_siamese_tensors(embedding_list_0, embedding_list_1, siamese_target):
    """
    Create pairing of all possible mode combinations of siamese pairs.  For example, if there were two modes,
    then the pairs would be `(elements_0_mode_0, elements_1_mode_0), (elements_0_mode_0, elements_1_mode_1),
    (elements_0_mode_1, elements_1_mode_0), (elements_0_mode_1, elements_1_mode_1)`. This increases the amount
    of training pairs for the generator.
    :param embedding_list_0: list of generator output batches for the first siamese pairs. Each batch tensor in the list
    corresponds to the generator outputs for batch element instances in a specific mode.
    :type: list<torch.Tensor>
    :param embedding_list_1: list of generator output batches for the second siamese pairs. Each batch tensor in the
    list corresponds to the generator outputs for batch element instances in a specific mode. The mode order must match
    that of `embedding_list_0`.
    :type: list<torch.Tensor>
    :param siamese_target: siamese target tensor for batch elements: tensor of length `batch_size` containing `0` if
    siamese pairs in an index have the same class (similar pair) or `1` otherwise (dissimilar pair).
    :type: torch.Tensor
    :return: tuple with the siamese pair and target tensors for all possible mode combinations of multimodal batch
    embeddings.
    :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
    """
    embedding_size = embedding_list_0[0].shape[1]
    siamese_pair_0 = torch.stack(embedding_list_0).repeat_interleave(2, 0).view(-1, embedding_size)
    siamese_pair_1 = torch.stack(embedding_list_1).repeat(2, 1, 1).view(-1, embedding_size)
    siamese_target = siamese_target.repeat(4)
    return siamese_pair_0, siamese_pair_1, siamese_target


def create_multimodal_gan_trainer(
        generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn, device=None,
        non_blocking=False, prepare_batch=_prepare_batch, prepare_batch_variables=prepare_multimodal_batch_variables,
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
    :param prepare_batch_variables: function that computes batch-derived variables needed for multimodal GAN processing:
    `classes`, `mode_labels`, `generator_labels`.
    :type: Callable<args: `batch`, `device`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a multimodal GAN trainer engine and returns the value
    to be assigned to engine's state.output after each iteration.
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
        classes, mode_labels, generator_labels = prepare_batch_variables(batch, device)

        ############################
        # (1) Update G network
        ###########################

        generator.zero_grad()
        embedding_list = generator(*[sub_batch[0] for sub_batch in batch])  # forward pass with same mode sub-batches
        embeddings = torch.cat(embedding_list)  # create a single discriminator batch from sub-batch list
        generator_loss = loss_fn(discriminator(embeddings), generator_labels)
        generator_loss.backward()
        generator_optimizer.step()

        ############################
        # (2) Update D network
        ###########################

        discriminator.zero_grad()
        mode_predictions = discriminator(embeddings.detach())
        discriminator_loss = loss_fn(mode_predictions, mode_labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        return output_transform(
            embeddings, mode_predictions, mode_labels, generator_labels, classes, generator_loss, discriminator_loss)

    return Engine(_update)


def create_multimodal_gan_siamese_trainer(
        generator, discriminator, generator_optimizer, discriminator_optimizer, mode_loss_fn,
        siamese_loss_fn, device=None, non_blocking=False, prepare_batch=prepare_batch_siamese,
        prepare_batch_variables=prepare_multimodal_batch_variables, output_transform=output_transform_trainer_siamese):
    """
    Factory function for creating an ignite trainer Engine for a multimodal GAN with a contrastive loss term.
    This engine is pretty much the same as the [normal multimodal GAN engine](create_multimodal_gan_trainer), but
    receives paired multimodal batches to allow the use of a contrastive term in the loss function that helps capture
    relationships amongst the different classes in the resulting vector space.
    :param generator: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
    :type: torch.nn.Module
    :param generator_optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param discriminator_optimizer: the optimizer to be used for the discriminator model
    :type: torch.optim.Optimizer
    :param mode_loss_fn: the loss function for mode prediction.
    :type: torch.nn loss function
    :param siamese_loss_fn: the loss function for siamese pairs.
    :type: torch.nn loss function
    :param device: device type specification
    :type: str of torch.device (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch.Tensor, torch.Tensor>> (optional)
    :param prepare_batch_variables: function that computes batch-derived variables needed for multimodal GAN processing:
    `classes`, `mode_labels`, `generator_labels`.
    :type: Callable<args: `batch`, `device`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a multimodal siamese GAN trainer engine and returns
    the value to be assigned to engine's state.output after each iteration.
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

        elements_0, elements_1, siamese_target = prepare_batch(batch, device=device, non_blocking=non_blocking)
        classes_0, mode_labels_0, generator_labels_0 = prepare_batch_variables(elements_0, device)
        classes_1, mode_labels_1, generator_labels_1 = prepare_batch_variables(elements_1, device)

        ############################
        # (1) Update G network
        ###########################

        generator.zero_grad()
        embedding_list_0 = generator(*[sub_batch[0] for sub_batch in elements_0]) # forward pass same mode sub-batches
        embedding_list_1 = generator(*[sub_batch[0] for sub_batch in elements_1])
        embeddings = torch.cat([*embedding_list_0, *embedding_list_1]) # create a single discriminator batch to allow..
        # noinspection PyTypeChecker
        generator_labels = torch.cat([generator_labels_0, generator_labels_1]) # ..us to do a single forward pass
        mode_labels = torch.cat([mode_labels_0, mode_labels_1])

        siamese_pair_0, siamese_pair_1, siamese_target = prepare_bimodal_siamese_tensors(
            embedding_list_0, embedding_list_1, siamese_target)

        # Optimize network
        generator_loss = mode_loss_fn(discriminator(embeddings), generator_labels) + \
                         siamese_loss_fn(siamese_pair_0, siamese_pair_1, siamese_target)
        generator_loss.backward()
        generator_optimizer.step()

        ############################
        # (2) Update D network
        ###########################

        discriminator.zero_grad()
        mode_predictions = discriminator(embeddings.detach())
        discriminator_loss = mode_loss_fn(mode_predictions, mode_labels)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        return output_transform(siamese_pair_0, siamese_pair_1, siamese_target, mode_predictions, mode_labels,
                                generator_labels, generator_loss, discriminator_loss)

    return Engine(_update)


def create_multimodal_gan_evaluator(
        generator, discriminator, metrics=None, device=None, non_blocking=False, prepare_batch=_prepare_batch,
        output_transform=output_transform_evaluator, prepare_batch_variables=prepare_multimodal_batch_variables):
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
    :param prepare_batch_variables: function that computes batch-derived variables needed for multimodal GAN processing:
    `classes`, `mode_labels`, `generator_labels`.
    :type: Callable<args: `batch`, `device`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a multimodal GAN evaluator engine and returns the
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
        discriminator.eval()
        with torch.no_grad():
            batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
            classes, mode_labels, generator_labels = prepare_batch_variables(batch, device)
            embedding_list = generator(*[sub_batch[0] for sub_batch in batch]) # forward pass with same mode sub-batches
            embeddings = torch.cat(embedding_list)  # create a single discriminator batch from sub-batch list
            mode_predictions = discriminator(embeddings)
            return output_transform(embeddings, mode_predictions, mode_labels, generator_labels, classes)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine


def create_multimodal_gan_siamese_evaluator(
        generator, discriminator, metrics=None, device=None, non_blocking=False, prepare_batch=_prepare_batch,
        output_transform=output_transform_evaluator_siamese,prepare_batch_variables=prepare_multimodal_batch_variables):
    """
    Factory function for creating an evaluator for a multimodal GAN with a contrastive loss term.
    This engine is pretty much the same as the [normal multimodal GAN engine](create_multimodal_gan_evaluator), but
    receives paired multimodal batches to allow the use of a contrastive term in the loss function that helps capture
    relationships amongst the different classes in the resulting vector space.
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
    :param prepare_batch_variables: function that computes batch-derived variables needed for multimodal GAN processing:
    `classes`, `mode_labels`, `generator_labels`.
    :type: Callable<args: `batch`, `device`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :param output_transform: function that receives the result of a multimodal siamese GAN evaluator engine and returns
    the value to be assigned to engine's state.output after each iteration, which must fit that expected by the metrics.
    :type: Callable<args: `anchor_embeddings`, `positive_embeddings`, `negative_embeddings`,
                    ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
    :return: an evaluator engine with supervised inference function.
    :type: ignite.engine.Engine
    """
    if device:
        generator.to(device)
        discriminator.to(device)

    # noinspection DuplicatedCode
    def _inference(_, batch):
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            elements_0, elements_1, siamese_target = prepare_batch(batch, device=device, non_blocking=non_blocking)
            classes_0, mode_labels_0, generator_labels_0 = prepare_batch_variables(elements_0, device)
            classes_1, mode_labels_1, generator_labels_1 = prepare_batch_variables(elements_1, device)
            generator.zero_grad()
            embedding_list_0 = generator(*[sub_batch[0] for sub_batch in elements_0])
            embedding_list_1 = generator(*[sub_batch[0] for sub_batch in elements_1])
            embeddings = torch.cat([*embedding_list_0, *embedding_list_1])
            # noinspection PyTypeChecker
            generator_labels = torch.cat([generator_labels_0, generator_labels_1])
            mode_labels = torch.cat([mode_labels_0, mode_labels_1])
            siamese_pair_0, siamese_pair_1, siamese_target = prepare_bimodal_siamese_tensors(
                embedding_list_0, embedding_list_1, siamese_target)
            mode_predictions = discriminator(embeddings.detach())
            return output_transform(
                siamese_pair_0, siamese_pair_1, siamese_target, mode_predictions, mode_labels, generator_labels)

    engine = Engine(_inference)
    if metrics: attach_metrics(engine, metrics)
    return engine
