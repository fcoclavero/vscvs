from ignite.engine import Engine

from src.utils.data import prepare_batch


def create_csv_gan_trainer(generator, discriminator, optimizer, loss_fn,
                           device=None, non_blocking=False, prepare_batch=prepare_batch):
    """
    Factory function for creating an ignite trainer Engine for the CSV GAN model.
    :param generator: the generator model - generates vectors from images
    :type: torch.nn.Module
    :param discriminator: the discriminator model - classifies vectors as 'photo' or 'sketch'
    :type: torch.nn.Module
    :param optimizer: the optimizer to be used
    :type: torch.optim.Optimizer
    :param loss_fn: the loss function
    :type: torch.nn loss function
    :param device: device type specification
    :type: str (optional) (default: None)
    :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
    :type: bool (optional)
    :param prepare_batch: batch preparation logic
    :type: Callable (args:`batch`,`device`,`non_blocking`, ret:tuple(torch.Tensor,torch.Tensor) (optional)
    :return: a trainer engine with the update function
    :type: ignite.engine.Engine
    """
    if device:
        generator.to(device)
        discriminator.to(device)
        print(generator)
        print(discriminator)

    def _update(engine, batch):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-image batch
        generator.zero_grad()
        # trainer = create_supervised_trainer(net, optimizer, criterion, device=device, prepare_batch=prepare_batch)
        # Train with all-sketch batch
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        discriminator.zero_grad()
        optimizer.zero_grad()
        x, y = prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)