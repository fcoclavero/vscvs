import click


@click.group()
def train():
    """ Train a model. """
    pass


@train.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--train_test_split', prompt='Train/test split',
              help='proportion of the dataset that will be used for training.', default=.7)
@click.option('--train_validation_split', prompt='Train/validation split',
              help='proportion of the training set that will be used for training, not validating.', default=.8)
@click.option('--lr', prompt='Learning rate', help='Learning rate for Adam optimizer', default=2e-4)
@click.option('--momentum', prompt='Momentum', help='Momentum parameter for SGD optimizer.', default=.2)
@click.option('--batch_size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@click.option('--epochs', prompt='Number of epochs', help='The number of training epochs.', type=int)
@click.option('--resume', help='Epoch for checkpoint loading.', default=None)
def cnn(dataset_name, train_test_split, train_validation_split, lr, momentum, batch_size, workers,
        n_gpu, epochs, resume):
    from src.trainers.cnn import train_cnn
    click.echo('cnn - %s dataset' % dataset_name)
    train_cnn(
        dataset_name, train_test_split, train_validation_split, lr, momentum, batch_size, workers, n_gpu, epochs, resume
    )


@train.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice([
        n + '_triplets' for n in ['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches']
    ])
)
@click.option(
    '--vector_dimension', prompt='CVS dimensionality', help='Dimensionality of the vector space.', default=300
)
@click.option('--resume', help='Epoch for checkpoint loading.', default=None)
@click.option('--margin', prompt='Margin', help='The margin for the Triplet Loss.', default=.2)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--batch_size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--lr', prompt='Learning rate', help='Learning rate for Adam optimizer', default=2e-4)
@click.option('--beta1', prompt='Beta 1', help='Decay parameter for Adam optimizer.', default=.2)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@click.option('--epochs', prompt='Number of epochs', help='The number of training epochs.', type=int)
def triplet_cnn(dataset_name, vector_dimension, resume, margin, workers, batch_size, lr, beta1, n_gpu, epochs):
    from src.trainers.triplet_cnn import train_triplet_cnn
    click.echo('triplet cnn - %s dataset' % dataset_name)
    train_triplet_cnn(dataset_name, vector_dimension, resume=resume, margin=margin, workers=workers,
                      batch_size=batch_size, n_gpu=n_gpu, epochs=epochs, learning_rate=lr, beta1=beta1)


@train.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_mixed_batches', 'sketchy_test_mixed_batches'])
)
@click.option(
    '--vector_dimension', prompt='CVS dimensionality', help='Dimensionality of the common vector space.', default=300
)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--batch_size', prompt='Batch size', help='The batch size during training.', type=int)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@click.option('--epochs', prompt='Number of epochs', help='The number of training epochs.', type=int)
def cvs_gan(dataset_name, vector_dimension, workers, batch_size, n_gpu, epochs):
    from src.trainers.cvs_gan import train_cvs_gan
    click.echo('cvs gan - %s dataset' % dataset_name)
    train_cvs_gan(dataset_name, vector_dimension, workers, batch_size, n_gpu, epochs)