import click

from settings import DATA_SETS


@click.group()
def train():
    """ Train a model. """
    pass


@train.command()
@click.option("--workers", prompt="Data loader workers", help="The number of workers for the data loader.", default=4)
@click.option("--batch_size", prompt="Batch size", help="The batch size during training.", type=int)
@click.option("--n_gpu", prompt="Number of gpus", help="The number of GPUs available. Use 0 for CPU mode.", default=0)
@click.option("--epochs", prompt="Number of epochs", help="The number of training epochs.", type=int)
def sketchy_cnn(workers, batch_size, n_gpu, epochs):
    # from src.trainers.sketchy_cnn import train_sketchy_cnn
    from src.trainers.sketchy_cnn_ignite import train_sketchy_cnn
    click.echo('sketchy cnn')
    train_sketchy_cnn(workers, batch_size, n_gpu, epochs)


@train.command()
def cvs_gan():
    from src.trainers.cvs_gan import train_cvs_gan
    click.echo('sketchy cnn')
    train_cvs_gan(DATA_SETS['sketchy_test']['images'], DATA_SETS['sketchy_test']['dimensions'][0])