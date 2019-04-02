from ignite.engine import Engine


def create_csv_gan_trainer(generator, discriminator, image_loader, sketch_loader, optimizer, loss_fn, device=None):
    def update_model(trainer, batch):
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
    return Engine(update_model)