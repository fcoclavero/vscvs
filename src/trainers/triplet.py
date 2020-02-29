__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a Triplet Network architecture. """


from ignite.metrics import Loss
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torchvision.models import resnet50, resnext50_32x4d

from src.models import ConvolutionalNetwork, TripletNetwork
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.triplet import create_triplet_evaluator, create_triplet_trainer
from src.utils.collators import triplet_collate
from src.utils.decorators import kwargs_parameter_dict


def triplet(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a triplet architecture.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a triplet architecture
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for a triplet network.
        """
        def __init__(self, *args, anchor_network=None, positive_negative_network=None, margin=1.0, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param anchor_network: the model to be used for computing anchor image embeddings.
            :type: torch.nn.Module
            :param positive_negative_network: the model to be used for computing the embeddings for the positive (same
            class) and negative (different class) elements of each triplet. The positive and negative elements are
            assumed to be of the same mode, and their embedding network will share weights.
            :type: torch.nn.Module
            :param margin: parameter for the triplet loss, defining the acceptable threshold for considering the
            embeddings of two examples as dissimilar.
            :type: float
            :param kwargs: AbstractTrainer keyword arguments
            :type: dict
            """
            self.anchor_network = anchor_network
            self.positive_negative_network = positive_negative_network
            self.margin = margin
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return TripletNetwork(self.anchor_network, self.positive_negative_network)

        @property
        def loss(self):
            return nn.MarginRankingLoss(margin=self.margin, reduction='mean')

        @property
        def trainer_id(self):
            return 'Siamese{}'.format(self.architecture_model.__class__.__name__)

        def _create_data_loaders(self, train_validation_split, batch_size, workers, drop_last, collate_fn=None):
            return super().create_data_loader(train_validation_split, batch_size, workers, drop_last,
                                              collate_fn=triplet_collate(default_collate))

        def _create_evaluator_engine(self):
            return create_triplet_evaluator(self.model, metrics={'loss': Loss(self.loss)}, device=self.device)

        def _create_trainer_engine(self):
            return create_triplet_trainer(self.model, self.optimizer, self.loss, device=self.device)

    return Trainer


@kwargs_parameter_dict
def train_triplet_cnn(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet CNN architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    @triplet
    @optimizer_decorator
    class TripletTrainer(AbstractTrainer):
        pass
    trainer = TripletTrainer(*args, architecture_model=ConvolutionalNetwork(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnet(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet ResNet architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    @triplet
    @optimizer_decorator
    class TripletTrainer(AbstractTrainer):
        pass
    trainer = TripletTrainer(*args, architecture_model=resnet50(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnext(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet ResNext architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    @triplet
    @optimizer_decorator
    class TripletTrainer(AbstractTrainer):
        pass
    trainer = TripletTrainer(*args, architecture_model=resnext50_32x4d(), **kwargs)
    trainer.run()
