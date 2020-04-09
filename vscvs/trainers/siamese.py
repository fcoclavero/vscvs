__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a siamese network. """


from vscvs.loss_functions import ContrastiveLoss
from vscvs.metrics.contrastive import Accuracy, AverageDistances, Loss
from vscvs.models import CNNNormalized, ResNetNormalized, ResNextNormalized, SiameseNetwork
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.siamese import create_siamese_evaluator, create_siamese_trainer
from vscvs.decorators import kwargs_parameter_dict


def siamese(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a siamese architecture.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a siamese architecture
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for a siamese network.
        """
        def __init__(self, *args, embedding_network_1=None, embedding_network_2=None, margin=.2, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param embedding_network_1: the model to be used for the first branch of the siamese architecture.
            :type: torch.nn.Module
            :param embedding_network_2: the model to be used for the second branch of the siamese architecture.
            :type: torch.nn.Module
            :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the
            embeddings of two examples as dissimilar. Dissimilar image pairs will be pushed apart unless their distance
            is already greater than the margin. Similar sketch–image pairs will be pulled together in the feature space.
            :type: float
            :param kwargs: AbstractTrainer keyword arguments
            :type: dict
            """
            self.embedding_network_1 = embedding_network_1
            self.embedding_network_2 = embedding_network_2
            self.margin = margin
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return SiameseNetwork(self.embedding_network_1, self.embedding_network_2)

        @property
        def loss(self):
            return ContrastiveLoss(margin=self.margin, reduction='mean')

        @property
        def trainer_id(self):
            return 'Siamese{}'.format(self.embedding_network_1.__class__.__name__)

        def _create_evaluator_engine(self):
            average_distances = AverageDistances()
            return create_siamese_evaluator(self.model, device=self.device, metrics={
                'accuracy': Accuracy(), 'average_positive_distance': average_distances[0],
                'average_negative_distance': average_distances[1], 'loss': Loss(self.loss)})

        def _create_trainer_engine(self):
            return create_siamese_trainer(self.model, self.optimizer, self.loss, device=self.device)

    return Trainer


@kwargs_parameter_dict
def train_siamese_cnn(*args, margin=.2, optimizer_mixin=None, **kwargs):
    """
    Train a Siamese CNN architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    class SiameseTrainer(optimizer_mixin, AbstractTrainer):
        pass

    trainer = SiameseTrainer(*args, embedding_network_1=CNNNormalized(out_features=250),  # photos
                             embedding_network_2=CNNNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnet(*args, margin=.2, optimizer_mixin=None, **kwargs):
    """
    Train a Siamese ResNet architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    class SiameseTrainer(optimizer_mixin, AbstractTrainer):
        pass

    trainer = SiameseTrainer(*args, embedding_network_1=ResNetNormalized(out_features=250, pretrained=True), # photos
                             embedding_network_2=ResNetNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnext(*args, margin=.2, optimizer_mixin=None, **kwargs):
    """
    Train a Siamese ResNext architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    class SiameseTrainer(optimizer_mixin, AbstractTrainer):
        pass

    trainer = SiameseTrainer(*args, embedding_network_1=ResNextNormalized(out_features=250, pretrained=True),  # photos
                             embedding_network_2=ResNextNormalized(out_features=250), margin=margin, **kwargs)
    trainer.run()
