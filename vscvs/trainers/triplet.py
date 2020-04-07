__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a Triplet Network architecture. """


from torch.utils.data._utils.collate import default_collate

from vscvs.loss_functions import TripletLoss
from vscvs.metrics.triplet import Accuracy, AverageDistances, Loss
from vscvs.models import CNNNormalized, ResNetNormalized, ResNextNormalized, TripletNetwork
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.triplet import create_triplet_evaluator, create_triplet_trainer
from vscvs.utils.collators import triplet_collate
from vscvs.decorators import kwargs_parameter_dict


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
        def __init__(self, *args, anchor_network=None, positive_negative_network=None, margin=.2, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param anchor_network: the model to be used for computing anchor image embeddings.
            :type: torch.nn.Module
            :param positive_negative_network: the model to be used for computing image embeddings for the positive
            and negative elements in each triplet.
            :type: torch.nn.Module
            :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
            distance from the anchor element to the negative, and the distance from the anchor to the negative.
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
            return TripletLoss(margin=self.margin, reduction='mean')

        @property
        def trainer_id(self):
            return 'Triplet{}'.format(self.anchor_network.__class__.__name__)

        def _create_data_loaders(self, train_validation_split, batch_size, workers, drop_last, collate_fn=None):
            return super().create_data_loader(train_validation_split, batch_size, workers, drop_last,
                                              collate_fn=triplet_collate(default_collate))

        def _create_evaluator_engine(self):
            average_distances = AverageDistances()
            return create_triplet_evaluator(self.model, device=self.device, metrics={
                'accuracy': Accuracy(), 'average_positive_distance': average_distances[0],
                'average_negative_distance': average_distances[1], 'loss': Loss(self.loss)})

        def _create_trainer_engine(self):
            return create_triplet_trainer(self.model, self.optimizer, self.loss, device=self.device)

    return Trainer


@kwargs_parameter_dict
def train_triplet_cnn(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet CNN architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
    distance from the anchor element to the negative, and the distance from the anchor to the negative.
    :type: float
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

    trainer = TripletTrainer(*args, anchor_network=CNNNormalized(out_features=250),
                             positive_negative_network=CNNNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnet(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet ResNet architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
    distance from the anchor element to the negative, and the distance from the anchor to the negative.
    :type: float
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

    trainer = TripletTrainer(*args, anchor_network=ResNetNormalized(out_features=250, pretrained=True),
                             positive_negative_network=ResNetNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnext(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Triplet ResNext architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
    distance from the anchor element to the negative, and the distance from the anchor to the negative.
    :type: float
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

    trainer = TripletTrainer(*args, anchor_network=ResNextNormalized(out_features=250, pretrained=True),
                             positive_negative_network=ResNextNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()
