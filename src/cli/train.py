__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Model training entry point. """


import click

from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context


@click.group()
@click.option('--resume', help='Epoch for checkpoint loading.', default=None)
@click.option('--train-validation-split', prompt='Train/validation split',
              help='proportion of the training set that will be used for training.', default=.8)
@click.option('--batch-size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--epochs', prompt='Number of epochs', help='The number of training epochs.', type=int)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def train(context, **kwargs):
    """ Train a model. """
    pass


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--lr', prompt='Learning rate', help='Learning rate for Adam optimizer', default=2e-4)
@click.option('--momentum', prompt='Momentum', help='Momentum parameter for SGD optimizer.', default=.2)
def cnn(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name, lr, momentum):
    from src.trainers.cnn import train_cnn
    click.echo('cnn - %s dataset' % dataset_name)
    train_cnn(dataset_name, train_validation_split, resume, batch_size, epochs, workers, n_gpu, lr, momentum)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-ketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option(
    '--vector-dimension', prompt='CVS dimensionality', help='Dimensionality of the vector space.', default=300
)
@click.option('--resume', help='Epoch for checkpoint loading.', default=None)
@click.option('--margin', prompt='Margin', help='The margin for the Triplet Loss.', default=.2)
@click.option('--lr', prompt='Learning rate', help='Learning rate for Adam optimizer', default=2e-4)
@click.option('--beta1', prompt='Beta 1', help='Decay parameter for Adam optimizer.', default=.2)
def triplet_cnn(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name, vector_dimension,
                margin, lr, beta1):
    from src.trainers.triplet_cnn import train_triplet_cnn
    click.echo('triplet cnn - %s dataset' % dataset_name)
    dataset_name = dataset_name + '-triplets'
    train_triplet_cnn(dataset_name, vector_dimension, resume, margin, workers, batch_size, n_gpu, epochs, lr, beta1)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-mixed-batches', 'sketchy-test-mixed-batches'])
)
@click.option(
    '--vector-dimension', prompt='CVS dimensionality', help='Dimensionality of the common vector space.', default=300
)
def cvs_gan(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name, vector_dimension):
    from src.trainers.cvs_gan import train_cvs_gan
    click.echo('cvs gan - %s dataset' % dataset_name)
    train_cvs_gan(dataset_name, vector_dimension, workers, batch_size, n_gpu, epochs)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-ketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option(
    '--vector-dimension', prompt='CVS dimensionality', help='Dimensionality of the vector space.', default=300
)
@click.option('--lr', prompt='Learning rate', help='Learning rate for Adam optimizer', default=2e-4)
@click.option('--weight-decay', prompt='Weight decay', help='Weight decay parameter for Adam optimizer.', default=5e-4)
def class_only_gcn(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name,
                   vector_dimension, lr, weight_decay):
    from src.trainers.class_only_gcn import train_class_only_gcn
    click.echo('class only GCN - %s dataset' % dataset_name)
    train_class_only_gcn(
        dataset_name, train_validation_split, resume, batch_size, epochs, workers, n_gpu, lr, weight_decay)
