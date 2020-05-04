__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a Triplet Network architecture. """


from abc import ABC

from vscvs.loss_functions import TripletLoss
from vscvs.metrics.triplet import Accuracy, AverageDistances, Loss
from vscvs.models import CNNNormalized, ResNetNormalized, ResNextNormalized, TripletSharedPositiveNegative
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.triplet import create_triplet_evaluator, create_triplet_trainer
from vscvs.decorators import kwargs_parameter_dict


class AbstractTripletTrainer(AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a triplet architecture.
    """
    def __init__(self, *args, anchor_network=None, positive_negative_network=None, loss_reduction='mean',
                 margin=.2, **kwargs):
        """
        :param args: Trainer arguments
        :type: tuple
        :param anchor_network: the model to be used for computing anchor image embeddings.
        :type: torch.nn.Module
        :param positive_negative_network: the model to be used for computing image embeddings for the positive
        and negative elements in each triplet.
        :type: torch.nn.Module
        :param loss_reduction: reduction to apply to batch element loss values to obtain the loss for the whole batch.
`       Must correspond to a valid reduction for the `TripletLoss`.
        :type: str
        :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
        distance from the anchor element to the negative, and the distance from the anchor to the negative.
        :type: float
        :param kwargs: Trainer keyword arguments
        :type: dict
        """
        self.anchor_network = anchor_network
        self.positive_negative_network = positive_negative_network
        self.loss_reduction = loss_reduction
        self.margin = margin
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        return TripletSharedPositiveNegative(self.anchor_network, self.positive_negative_network)

    @property
    def loss(self):
        return TripletLoss(margin=self.margin, reduction=self.loss_reduction)

    @property
    def trainer_id(self):
        return 'Triplet{}'.format(self.anchor_network.__class__.__name__)

    def _create_evaluator_engine(self):
        average_distances = AverageDistances()
        return create_triplet_evaluator(self.model, device=self.device, metrics={
            'accuracy': Accuracy(), 'average_positive_distance': average_distances[0],
            'average_negative_distance': average_distances[1], 'loss': Loss(self.loss)})

    def _create_trainer_engine(self):
        return create_triplet_trainer(self.model, self.optimizer, self.loss, device=self.device)


@kwargs_parameter_dict
def train_triplet_cnn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet CNN architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        pass
    trainer = TripletTrainer(*args, anchor_network=CNNNormalized(out_features=250),
                             positive_negative_network=CNNNormalized(out_features=250), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnet(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet ResNet architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        pass
    trainer = TripletTrainer(*args, anchor_network=ResNetNormalized(out_features=250, pretrained=True),
                             positive_negative_network=ResNetNormalized(out_features=250), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnext(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet ResNext architecture.
    :param args: TripletTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: dict
    """
    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        pass
    trainer = TripletTrainer(*args, anchor_network=ResNextNormalized(out_features=250, pretrained=True),
                             positive_negative_network=ResNextNormalized(out_features=250), **kwargs)
    trainer.run()
