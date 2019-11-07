__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN over image labels. """


from torch.nn import NLLLoss
from torch.optim import Adam

from src.models.graph_convolutional_network import GCN
from src.trainers.abstract_trainer import AbstractTrainer


class ClassOnlyGCNTrainer(AbstractTrainer):
    """
    Trainer for a class classification GCN that uses only image classes and batch clique graphs where vertex weights
    correspond to word vector distances between image class labels.
    """
    def __init__(self, dataset_name, train_validation_split=.8, resume_checkpoint=None, batch_size=16, workers=4,
                 n_gpu=0, epochs=2, learning_rate=.01, weight_decay=5e-4):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        super().__init__(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu, epochs)

    @property
    def initial_model(self):
        return GCN()

    @property
    def loss(self):
        return NLLLoss()

    @property
    def optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay}

    @property
    def trainer_id(self):
        return 'class_only_gcn'

    def _create_evaluator(self):
        pass

    def _create_trainer_engine(self):
        pass


def train_class_only_gcn(dataset_name, train_validation_split=.8, resume_checkpoint=None, batch_size=16, epochs=2,
                         workers=4, n_gpu=0, learning_rate=.01, weight_decay=5e-4):
    """
    Trains a GCN to predict image labels using a GCN over batch clique graphs where nodes correspond to batch images and
    vertex weights corresponds to image label word vector distances.
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
    :param weight_decay: weight_decay parameter for Adam optimizer
    :type: float
    """
    trainer = ClassOnlyGCNTrainer(dataset_name, train_validation_split, resume_checkpoint, batch_size, workers, n_gpu,
                                  epochs, learning_rate, weight_decay)
    trainer.run()
