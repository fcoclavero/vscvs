__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Abstract class with the basic boilerplate code needed to define and run Ignite engines. """


import os
import torch

from abc import ABC, abstractmethod
from datetime import datetime
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, TerminateOnNan, Timer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import CHECKPOINT_NAME_FORMAT
from src.datasets import get_dataset
from src.utils import get_device, get_checkpoint_directory, get_log_directory
from src.utils.data import dataset_split_successive


class AbstractTrainer(ABC):
    """
    Abstract class with the boilerplate code needed to define and run an Ignite trainer Engine.
    """
    def __init__(self, *args, batch_size=0, dataset_name=None, drop_last=False, epochs=1, n_gpu=0, parameter_dict=None,
                 resume_date=None, resume_checkpoint=None, tag=None, train_validation_split=.8, workers=6, **kwargs):
        """
        Base constructor which sets default trainer parameters.
        :param args: mixin arguments
        :type: tuple
        :param batch_size: batch size during training
        :type: int
        :param dataset_name: the name of the Dataset to be used for training
        :type: str
        :param drop_last: whether to drop the last batch if it is not the same size as `batch_size`.
        :type: boolean
        :param epochs: the number of epochs used for training
        :type: int
        :param n_gpu: number of GPUs available. Use 0 for CPU mode
        :type: int
        :param parameter_dict: dictionary with important training parameters for logging.
        :type: dict
        :param resume_date: date of the trainer state to be resumed. Dates must have this format: `%y-%m-%dT%H-%M`
        :type: str
        :param resume_checkpoint: name of the model checkpoint to be loaded.
        :type: str
        :param tag: optional tag for model checkpoint and tensorboard logs
        :type: int
        :param train_validation_split: proportion of the training set that will be used for actual
        training. The remaining data will be used as the validation set.
        :type: float
        :param workers: number of workers for data_loader
        :type: int
        :param kwargs: mixin keyword arguments
        :type: dict
        """
        date = datetime.now()
        self.batch_size = batch_size
        self.checkpoint_directory = get_checkpoint_directory(self.trainer_id, tag=tag, date=date)
        self.dataset = get_dataset(dataset_name)
        self.dataset_name = dataset_name
        self.device = get_device(n_gpu)
        self.epochs = epochs
        self.log_directory = get_log_directory(self.trainer_id, tag=tag, date=date)
        self.model = self.initial_model.to(self.device)
        self.parameter_dict = parameter_dict
        self.resume_date = datetime.strptime(resume_date, CHECKPOINT_NAME_FORMAT) if resume_date else resume_date
        self.resume_checkpoint = resume_checkpoint
        self.start_epoch = 1
        self.tag = tag
        self._load_checkpoint()
        self.epoch = self.start_epoch
        self.step = 0
        self.train_loader, self.validation_loader = \
            self._create_data_loaders(train_validation_split, batch_size, workers, drop_last)
        self.trainer_engine = self._create_trainer_engine()
        self.evaluator_engine = self._create_evaluator_engine()
        self.timer = self._create_timer()
        self._add_event_handlers()
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def initial_model(self):
        """
        Getter for an untrained nn.module for the model the Trainer is designed for. Used to initialize `self.model`.
        :return: a model object
        :type: torch.nn.module
        """
        pass

    @property
    @abstractmethod
    def loss(self):
        """
        Getter for the loss to be used during training.
        :return: a loss object
        :type: torch.nn._Loss
        """
        pass

    @property
    @abstractmethod
    def optimizer(self):
        """
        Getter for the optimizer to be used during training.
        :return: an optimizer object
        :type: torch.optim.Optimizer
        """
        pass

    @property
    def trainer_checkpoint(self):
        """
        Getter for the serialized checkpoint dictionary, which contains the values of the trainer's fields that should
        be saved in a trainer checkpoint.
        :return: a checkpoint dictionary
        :type: dict
        """
        return {
            'average_epoch_duration': self.timer.value(),
            'batch_size': self.batch_size,
            'dataset_name': self.dataset_name,
            'parameters': self.parameter_dict,
            'resume_date': self.resume_date,
            'resume_checkpoint': self.resume_checkpoint,
            'start_epoch': self.start_epoch,
            'epochs': self.epochs,
            'tag': self.tag
        }

    @property
    @abstractmethod
    def trainer_id(self):
        """
        Getter for the trainer id, a unique str to identify the trainer. The corresponding `data` directory sub-folders
        will get a name containing this id.
        :return: the trainer id
        :type: str
        """
        pass

    def _add_event_handlers(self):
        """
        Adds a progressbar and a summary writer to output the current training status. Adds event handlers to output
        common messages and update the progressbar.
        """
        progressbar_description = 'TRAINING => loss: {:.6f}'
        progressbar = tqdm(initial=0, leave=False, total=len(self.train_loader), desc=progressbar_description.format(0))
        writer = SummaryWriter(self.log_directory)

        @self.trainer_engine.on(Events.ITERATION_COMPLETED)
        def update_step_counter(trainer):
            self.step += 1

        @self.trainer_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            writer.add_scalar('training_loss', trainer.state.output, self.step)
            progressbar.desc = progressbar_description.format(trainer.state.output)
            progressbar.update(1)

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def update_epoch_counter(trainer):
            self.epoch += 1

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            self.evaluator_engine.run(self.train_loader)
            metrics = self.evaluator_engine.state.metrics
            print('\nTraining results - epoch: {}'.format(self.epoch))
            for key, value in metrics.items():
                writer.add_scalar('training_{}'.format(key), value, self.step)
                print('{}: {:.6f}'.format(key, value))

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.evaluator_engine.run(self.validation_loader)
            metrics = self.evaluator_engine.state.metrics
            print('\nValidation results - epoch: {}'.format(self.epoch))
            for key, value in metrics.items():
                writer.add_scalar('validation_{}'.format(key), value, self.step)
                print('{}: {:.6f}'.format(key, value))

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def update_trainer_checkpoint(trainer):
            self._save_trainer_checkpoint()

        @self.trainer_engine.on(Events.EPOCH_COMPLETED)
        def reset_progressbar(trainer):
            progressbar.n = progressbar.last_print_n = 0
            progressbar.reset(total=len(self.train_loader))

        @self.trainer_engine.on(Events.COMPLETED)
        def cleanup(trainer):
            writer.close()
            progressbar.close()

        periodic_checkpoint_saver = ModelCheckpoint( # create a Checkpoint handler that can be used to periodically
            self.checkpoint_directory, filename_prefix='net_latest', # save model objects to disc.
            save_interval=1, n_saved=3, atomic=True, create_dir=True, save_as_state_dict=True, require_empty=False
        )
        best_checkpoint_saver = ModelCheckpoint( # create a Checkpoint handler that can be used to save the best
            self.checkpoint_directory, filename_prefix='net_best', # performing models
            save_interval=1, n_saved=5, atomic=True, create_dir=True, save_as_state_dict=True, require_empty=False
        )
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint_saver, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, periodic_checkpoint_saver, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.COMPLETED, periodic_checkpoint_saver, {'complete': self.model})

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

    @abstractmethod
    def _create_evaluator_engine(self):
        """
        Creates an Ignite evaluator engine for the target model.
        :return: an evaluator engine for the target model
        :type: ignite.Engine
        """
        pass

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

    @abstractmethod
    def _create_trainer_engine(self):
        """
        Creates an Ignite training engine for the target model.
        :return: a trainer engine for the target model
        :type: ignite.Engine
        """
        pass

    def _load_checkpoint(self):
        """
        Load the state and model checkpoints and update the trainer to continue training.
        """
        if self.resume_date:
            try:
                previous_checkpoint_directory = \
                    get_checkpoint_directory(self.trainer_id, tag=self.tag, date=self.resume_date)
            except FileNotFoundError:
                raise FileNotFoundError('Checkpoint {} not found.'.format(self.resume_date))
            self._load_model_checkpoint(previous_checkpoint_directory)
            self._load_trainer_checkpoint(previous_checkpoint_directory)
            tqdm.write('Successfully loaded the {} checkpoint.'.format(self.resume_date))

    def _load_model_checkpoint(self, previous_checkpoint_directory):
        """
        Load the model state_dict saved in the model checkpoint file into the already initialized model field.
        :param previous_checkpoint_directory: directory containing the checkpoint to me loaded.
        :type: str
        """
        state_dict = torch.load(os.path.join(previous_checkpoint_directory, '{}.pth'.format(self.resume_checkpoint)))
        self.model.load_state_dict(state_dict)

    def _load_trainer_checkpoint(self, previous_checkpoint_directory):
        """
        Load trainer fields from the specified saved trainer checkpoint file.
        :param previous_checkpoint_directory: directory containing the checkpoint to me loaded.
        :type: str
        """
        previous_state = torch.load(os.path.join(previous_checkpoint_directory, 'trainer.pth'))
        self.start_epoch = previous_state['start_epoch'] + previous_state['epochs']

    def _save_trainer_checkpoint(self):
        """
        Create the serialized checkpoint dictionary for the current trainer state, and save it.
        """
        torch.save(self.trainer_checkpoint, os.path.join(self.checkpoint_directory, 'trainer.pth'))

    def run(self):
        """
        Run the trainer.
        """
        self.trainer_engine.run(self.train_loader, max_epochs=self.epochs)
