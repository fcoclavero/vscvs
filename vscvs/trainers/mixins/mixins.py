__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Mixins with the code needed to add different functionality to Trainers. """


from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping


class EarlyStoppingMixin:
    """
    Mixin class for adding early stopping to a Trainer. The mixin must be inherited after the AbstractTrainer class in
    order to have access to the Trainer's `evaluator_engine` and `trainer_engine`.
    """
    evaluator_engine: Engine
    trainer_engine: Engine

    def __init__(self, *args, early_stopping_patience=5, **kwargs):
        """
        Mixin constructor which creates and attaches an EarlyStopping handler to the Trainer.
        :param args: arguments for additional mixin
        :type: tuple
        :param early_stopping_patience: number of epochs to wait if there are no improvements to stop the training.
        :type: int
        :param kwargs: keyword arguments for additional mixin
        :type: dict
        """
        self.early_stopping_patience = early_stopping_patience
        self.evaluator_engine.add_event_handler(Events.COMPLETED, self._early_stopping_handler)
        super().__init__(*args, **kwargs)

    @property
    def _early_stopping_handler(self):
        """
        Create the EarlyStopping handler that will evaluate the `score_function` class on each `evaluator_engine` run
        and stop the `trainer_engine` if there has been no improvement in the `_score_function` for the number of
        epochs specified in `early_stopping_patience`.
        :return: the early stopping handler
        :type: from ignite.handlers.EarlyStopping
        """
        return EarlyStopping(patience=self.early_stopping_patience, score_function=self._score_function,
                             trainer=self.trainer_engine)

    @staticmethod
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
        raise NotImplementedError
