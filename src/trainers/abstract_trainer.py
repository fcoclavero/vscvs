__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Abstract class and mixins with the boilerplate code needed to define and run Ignite engines. """


import os
import torch

from datetime import datetime
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Timer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device, get_checkpoint_directory, get_log_directory
from src.utils.data import dataset_split_successive


class AbstractTrainer:
    """
    Abstract class with the boilerplate code needed to define and run an Ignite trainer Engine.
    """
    def __init__(self, *args, dataset_name=None,  resume_date=None, train_validation_split=.8, batch_size=16, epochs=2,
                 workers=6, n_gpu=0, tag=None, drop_last=False, parameter_dict=None, **kwargs):
        """
        Base constructor which sets default trainer parameters.
        :param args: mixin arguments
        :type: tuple
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
        :type: int
        :param drop_last: whether to drop the last batch if it is not the same size as `batch_size`.
        :type: boolean
        :param parameter_dict: dictionary with important training parameters for logging.
        :type: dict
        :param kwargs: mixin keyword arguments
        :type: dict
        """
        date = resume_date or datetime.now()
        self.batch_size = batch_size
        self.checkpoint_directory = get_checkpoint_directory(self.trainer_id, tag=tag, date=date)
        self.dataset = get_dataset(dataset_name)
        self.dataset_name = dataset_name
        self.device = get_device(n_gpu)
        self.epochs = epochs
        self.event_handlers = []
        self.log_directory = get_log_directory(self.trainer_id, tag=tag, date=date)
        self.model = self.initial_model.to(self.device)
        self.parameter_dict = parameter_dict
        self.start_epoch = 0
        self.steps = 0
        self.train_loader, self.validation_loader = \
            self._create_data_loaders(train_validation_split, batch_size, workers, drop_last)
        self.trainer_engine = self._create_trainer_engine()
        self.evaluator_engine = self._create_evaluator_engine()
        self.timer = self._create_timer()
        self._add_event_handlers()
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        """
        Getter for an untrained nn.module for the model the Trainer is designed for. Used to initialize `self.model`.
        :return: a model object
        :type: torch.nn.module
        """
        raise NotImplementedError

    @property
    def loss(self):
        """
        Getter for the loss to be used during training.
        :return: a loss object
        :type: torch.nn._Loss
        """
        raise NotImplementedError

    @property
    def optimizer(self):
        """
        Getter for the optimizer to be used during training.
        :return: an optimizer object
        :type: torch.optim.Optimizer
        """
        raise NotImplementedError

    @property
    def serialized_checkpoint(self):
        """
        Getter for the serialized checkpoint dictionary, which contains the values of the trainer's fields that should
        be saved in a trainer checkpoint.
        :return: a checkpoint dictionary
        :type: dict
        """
        return {
            'average_epoch_duration': self.timer.value(),
            'dataset_name': self.dataset_name,
            'last_run': datetime.now(),
            'optimizer': self.optimizer,
            'parameters': self.parameter_dict,
            'total_epochs': self.start_epoch + self.epochs
        }

    @property
    def trainer_id(self):
        """
        Getter for the trainer id, a unique str to identify the trainer. The corresponding `data` directory sub-folders
        will get a name containing this id.
        :return: the trainer id
        :type: str
        """
        raise NotImplementedError

    def _add_event_handlers(self):
        """
        Adds a progressbar and a summary writer to output the current training status. Adds event handlers to output
        common messages and update the progressbar.
        """
        progressbar_description = 'TRAINING => loss: {:.6f}'
        progressbar = tqdm(initial=0, leave=False, total=len(self.train_loader), desc=progressbar_description.format(0))
        writer = SummaryWriter(self.log_directory)

        @self.trainer_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            writer.add_scalar('training_loss', trainer.state.output, self.steps)
            progressbar.desc = progressbar_description.format(trainer.state.output)
            progressbar.update(1)
            self.steps += 1

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            self.evaluator_engine.run(self.train_loader)
            metrics = self.evaluator_engine.state.metrics
            print('\nTraining results - epoch: {}'.format(trainer.state.epoch))
            for key, value in metrics.items():
                writer.add_scalar('training_{}'.format(key), value, self.steps)
                print('{}: {:.6f}'.format(key, value))

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.evaluator_engine.run(self.validation_loader)
            metrics = self.evaluator_engine.state.metrics
            print('\nValidation results - epoch: {}'.format(trainer.state.epoch))
            for key, value in metrics.items():
                writer.add_scalar('validation_{}'.format(key), value, self.steps)
                print('{}: {:.6f}'.format(key, value))

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def reset_progressbar(trainer):
            progressbar.n = progressbar.last_print_n = 0
            progressbar.reset(total=len(self.train_loader))

        @self.trainer_engine.on(Events.COMPLETED)
        def cleanup(trainer):
            writer.close()
            progressbar.close()

        checkpoint_saver = ModelCheckpoint( # create a Checkpoint handler that can be used to periodically
            self.checkpoint_directory, filename_prefix='net', # save model objects to disc.
            save_interval=1, n_saved=5, atomic=True, create_dir=True, save_as_state_dict=True, require_empty=False
        )
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.COMPLETED, checkpoint_saver, {'complete': self.model})

    def _create_data_loaders(self, train_validation_split, batch_size, workers, drop_last):
        """
        Create training and validation data loaders, placing a total of `len(self.dataset) * train_validation_split`
        elements in the training subset.
        :param train_validation_split: the proportion of dataset elements to be placed in the training subset.
        :type: float $\in [0, 1]$
        :param batch_size: batch size for both data loaders.
        :type: int
        :param workers: number of workers for both data loaders.
        :type: int
        :param drop_last: if `True`, the last batch will be dropped if incomplete (when the dataset size is not
        divisible by the batch size). If `False` and the dataset size is not divisible by the batch size, the last
        batch will have a smaller size than the rest.
        :type: bool
        :return: two DataLoaders, the first for the training data and the second for the validation data.
        :type: torch.utils.data.DataLoader
        """
        loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last)
                   for subset in dataset_split_successive(self.dataset, train_validation_split)]
        if not len(loaders[-1]):
            raise ValueError('Empty validation loader. This might be caused by having `drop_last=True` and \
                             a resulting validation set smaller than `batch_size`.')
        return loaders

    def _create_evaluator_engine(self):
        """
        Creates an Ignite evaluator engine for the target model.
        :return: an evaluator engine for the target model
        :type: ignite.Engine
        """
        raise NotImplementedError

    def _create_timer(self):
        """
        Create and attach a new timer to the trainer, registering callbacks.
        :return: the newly created timer
        :type: ignite.handlers.Timer
        """
        timer = Timer(average=True)
        timer.attach(self.trainer_engine, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        return timer

    def _create_trainer_engine(self):
        """
        Creates an Ignite training engine for the target model.
        :return: a trainer engine for the target model
        :type: ignite.Engine
        """
        raise NotImplementedError

    def _deserialize_checkpoint(self, checkpoint):
        """
        Load trainer fields from a serialized checkpoint dictionary.
        :param checkpoint: the checkpoint being loaded
        :type: dict
        """
        self.start_epoch = checkpoint['epochs']
        self.model = torch.load(os.path.join(self.checkpoint_directory, '_model_{}.pth'.format(self.start_epoch)))

    def _load_checkpoint(self, resume_date):
        """
        Load the trainer checkpoint dictionary for the given resume date and deserialize it (load the values in the
        checkpoint dictionary into the trainer's fields).
        :param resume_date: checkpoint folder name containing model and checkpoint .pth files containing the information
        needed for resuming training. Folder names correspond to dates with the following format: `%y-%m-%dT%H-%M`
        :type: str
        """
        try:
            self._deserialize_checkpoint(torch.load(os.path.join(self.checkpoint_directory, 'checkpoint.pth')))
            tqdm.write('Successfully loaded the {} checkpoint.'.format(resume_date))
        except FileNotFoundError:
            raise FileNotFoundError('Checkpoint {} not found.'.format(resume_date))

    def _save_checkpoint(self):
        """
        Create the serialized checkpoint dictionary for the current trainer state, and save it.
        """
        torch.save(self.serialized_checkpoint, os.path.join(self.checkpoint_directory, 'checkpoint.pth'))

    def run(self):
        """
        Run the trainer.
        """
        self.trainer_engine.run(self.train_loader, max_epochs=self.epochs)


class EarlyStoppingMixin:
    """
    Mixin class for adding early stopping to a Trainer. The mixin must be inherited after the AbstractTrainer class in
    order to have access to the Trainer's `evaluator_engine` and `trainer_engine`.
    """
    def __init__(self, *args, early_stopping_patience=5, **kwargs):
        """
        Mixin constructor which creates and attaches an EarlyStopping handler to the Trainer.
        :param args: additional mixin arguments
        :type: tuple
        :param early_stopping_patience: number of epochs to wait if there are no improvements to stop the training.
        :type: int
        :param kwargs: additional mixin keyword arguments
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
