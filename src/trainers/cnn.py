__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a CNN classification network. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from src.models.convolutional.classification import ClassificationConvolutionalNetwork
from src.trainers.abstract_trainer import AbstractTrainer
from src.utils.data import prepare_batch


class CNNTrainer(AbstractTrainer):
    """
    Trainer for a simple class classification CNN.
    """
    def __init__(self, dataset_name, train_validation_split=.8, resume_checkpoint=None, batch_size=16, workers=4,
                 n_gpu=0, epochs=2, learning_rate=.01, momentum=.8):
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu, epochs)

    @property
    def initial_model(self):
        return ClassificationConvolutionalNetwork()

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

    @property
    def trainer_id(self):
        return 'cnn_sk'

    def _create_evaluator(self):
        return create_supervised_evaluator(
            self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


def train_cnn(dataset_name, train_validation_split=.8, resume_checkpoint=None, batch_size=16, epochs=2, workers=4,
              n_gpu=0, learning_rate=.01, momentum=.8):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    :param resume_checkpoint: date of the trainer state to be resumed. Dates must have the following
    format: `%y-%m-%dT%H-%M`
    :type: str
    :param batch_size: batch size during training
    :type: int
    :param workers: number of workers for data_loader
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :param learning_rate: learning rate for optimizers
    :type: float
    :param momentum: momentum parameter for SGD optimizer
    :type: float
    """
    trainer = CNNTrainer(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu, epochs,
                         learning_rate, momentum)
    trainer.run()
