import torch

from ignite.engine import Engine

from src.utils.initialize_weights import initialize_weights
from src.utils.data import prepare_batch


def create_csv_gan_trainer(generator, discriminator, generator_optimizer, discriminator_optimizer,
                           gan_loss, vector_dimension, photos_label=1, device=None,
                           non_blocking=False, prepare_batch=prepare_batch):
    """
    Factory function for creating an ignite trainer Engine for the CSV GAN model.
    NOTES:
        * The discriminator (D) output should be high if an image is a photo (and `photos_label=1`).
          That is, $D(photo)$ should be close to 1 and $D(sketch)$ should be close to 0.
        * The discriminator (D) tries to maximize the probability it correctly classifies photos
          and sketches. This can be expressed in two terms: $log(D(G(photo)))$ and $log(D(G(sketch)))$.
        * The generator (G) tries to minimize the probability that D will predict its outputs
          correctly: $log(1−D(G(sketch)))$ and $log(1−D(G(image)))$.
    :param generator: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
    :type: torch.nn.Module
    :param generator_optimizer: the optimizer to be used for the generator model
    :type: torch.optim.Optimizer
    :param discriminator_optimizer: the optimizer to be used for the discriminator model
    :type: torch.optim.Optimizer
    :param gan_loss: the loss function for the GAN model
    :type: torch.nn loss function
    :param vector_dimension: the dimensionality of the common vector space.
    :type: int
    :param photos_label: the binary label (1 or 0) to identify images as belonging to the photo
    modality. The label for sketches is deduced as the opposite of the given value.
    :param device: device type specification
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable (args:`batch`,`device`,`non_blocking`, ret:tuple(torch.Tensor,torch.Tensor) (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    sketches_label = int(not photos_label) # model is bi-modal for the time being

    print(generator)
    print(discriminator)

    if device:
        generator.to(device)
        discriminator.to(device)

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    def gan_forward_pass(images, labels):
        # Training mode
        generator.train()
        discriminator.train()
        # Forward pass image batch through the generator network
        vectors = generator(images).view(-1, vector_dimension)
        # Predict modality using the discriminator network
        prediction = discriminator(vectors).view(-1)
        # Calculate the generator loss: we use 1 - labels, as we want the generator to maximize
        generator_loss = gan_loss(prediction, 1 - labels) # discriminator mistakes
        # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True
        generator_loss.backward() # grads are accumulated into x.grad for every param. x (by addition)
        # Calculate the discriminator loss
        discriminator_loss = gan_loss(prediction, labels)
        # Accumulate gradients
        discriminator_loss.backward()
        # Return losses for logging
        return generator_loss.item(), discriminator_loss.item()

    def _update(engine, batch):
        photos, sketches, classes = batch
        # Create tensor of all sketches, no matter the class or associated photo, for CVS training
        all_sketches = torch.cat(sketches, 0)
        # Generate mode label arrays
        photo_labels = torch.full((photos.size(0),), photos_label, device=device)
        sketches_labels = torch.full((all_sketches.size(0),), sketches_label, device=device)
        # Reset gradients
        generator.zero_grad()
        discriminator.zero_grad()
        # Train with all-photo sub-batch
        photo_generator_loss, photo_discriminator_loss = gan_forward_pass(photos, photo_labels)
        # Train with all-sketch sub-batch
        sketch_generator_loss, sketch_discriminator_loss = gan_forward_pass(all_sketches, sketches_labels)
        # Add the losses from the all-photo and all-sketch sub-batches
        generator_loss = photo_generator_loss + sketch_generator_loss
        discriminator_loss = photo_discriminator_loss + sketch_discriminator_loss
        # Update model wights
        generator_optimizer.step()
        discriminator.step()
        # Return losses, for reasons
        return generator_loss, discriminator_loss

    return Engine(_update)