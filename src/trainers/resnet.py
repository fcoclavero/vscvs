__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a ResNet classification network. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet50

from src.trainers.abstract_trainer import AbstractTrainer
from src.utils.data import prepare_batch


class ResNetTrainer(AbstractTrainer):
    """
    Trainer for a ResNext image classifier.
    """
    def __init__(self, dataset_name, resume_date=None, train_validation_split=.8, batch_size=16, epochs=2, workers=6,
                 n_gpu=0, tag=None, learning_rate=.01, momentum=.8, drop_last=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(dataset_name, resume_date, train_validation_split, batch_size, epochs, workers, n_gpu, tag)

    @property
    def initial_model(self):
        return resnet50()

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
        return 'resnet'

    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


def train_resnet(dataset_name, resume_date=None, train_validation_split=.8, batch_size=16, epochs=2, workers=4,
                  n_gpu=0, tag=None, learning_rate=.01, momentum=.8):
    """
    Train a ResNet image classifier.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param resume_date: date of the trainer state to be resumed. Dates must have the following
    format: `%y-%m-%dT%H-%M`
    :type: str
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    :param batch_size: batch size during training
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :param workers: number of workers for data_loader
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param tag: optional tag for model checkpoint and tensorboard logs
    :type: str
    :param learning_rate: learning rate for optimizers
    :type: float
    :param momentum: momentum parameter for SGD optimizer
    :type: float
    """
    trainer = ResNetTrainer(dataset_name, resume_date, train_validation_split, batch_size, epochs, workers, n_gpu, tag,
                             learning_rate, momentum)
    trainer.run()
