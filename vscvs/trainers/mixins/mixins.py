__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Mixins with the code needed to add different functionality to Trainers. """


from abc import ABC
from abc import abstractmethod

from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import EarlyStopping


class EarlyStoppingMixin(ABC):
    """
    Mixin class for adding early stopping to a Trainer.
    """

    evaluator_engine: Engine
    trainer_engine: Engine

    def __init__(self, *args, early_stopping_patience=5, **kwargs):
        """
        :param args: arguments for additional mixin
        :type: Tuple
        :param early_stopping_patience: number of epochs to wait if there are no improvements to stop the training.
        If `None`, no early stopping will be applied.
        :type: int
        :param kwargs: keyword arguments for additional mixin
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        self.early_stopping_patience = early_stopping_patience
        if early_stopping_patience:
            self.evaluator_engine.add_event_handler(Events.COMPLETED, self._early_stopping_handler)

    @property
    def _early_stopping_handler(self):
        """
        Create the EarlyStopping handler that will evaluate the `score_function` class on each `evaluator_engine` run
        and stop the `trainer_engine` if there has been no improvement in the `_score_function` for the number of
        epochs specified in `early_stopping_patience`.
        :return: the early stopping handler
        :type: from ignite.handlers.EarlyStopping
        """
        return EarlyStopping(
            patience=self.early_stopping_patience, score_function=self._score_function, trainer=self.trainer_engine
        )

    @staticmethod
    @abstractmethod
    def _score_function(engine):
        """
        Function needed by the early stopping event handler that will receive the engine and must return a single
        score float. An improvement in the training is considered if the score returned by this function is higher
        than in previous training steps. The trainer engine will stop if there is no improvement in
        `self.early_stopping_patience` steps.
        :param engine: engine provided by the event handler
        :type: ignite.engine.Engine
        :return: a single float that is bigger as the model improves
        :type: float
        """
        pass
