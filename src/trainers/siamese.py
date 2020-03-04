__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a siamese network. """


from ignite.metrics import Loss

from src.loss_functions import ContrastiveLoss
from src.models import CNN, SiameseNetwork
from src.models import ResNet
from src.models import ResNext
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.siamese import create_siamese_evaluator, create_siamese_trainer
from src.utils.decorators import kwargs_parameter_dict


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
            return ContrastiveLoss(margin=self.margin)

        @property
        def trainer_id(self):
            return 'Siamese{}'.format(self.embedding_network_1.__class__.__name__)

        def _create_evaluator_engine(self):
            return create_siamese_evaluator(self.model, metrics={'loss': Loss(self.loss)}, device=self.device)

        def _create_trainer_engine(self):
            return create_siamese_trainer(self.model, self.optimizer, self.loss, device=self.device)

    return Trainer


@kwargs_parameter_dict
def train_siamese_cnn(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese CNN architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network_1=CNN(),  # photos
                             embedding_network_2=CNN(), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnet(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese ResNet architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network_1=ResNet(out_features=250, pretrained=True), # photos
                             embedding_network_2=ResNet(out_features=250), margin=margin, **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnext(*args, margin=.2, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese ResNext architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
    of two examples as dissimilar.
    :type: float
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network_1=ResNext(out_features=250, pretrained=True), # photos
                             embedding_network_2=ResNext(out_features=250), margin=margin, **kwargs)
    trainer.run()
