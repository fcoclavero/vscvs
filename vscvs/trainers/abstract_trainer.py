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
from vscvs.datasets import get_dataset
from vscvs.utils import get_device, get_checkpoint_directory, get_log_directory
from vscvs.utils.data import dataset_split_successive


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
        self.start_epoch = 0
        self.tag = tag
        self._load_checkpoint()
        self.epoch = self.start_epoch
        self.step = 0
        self.train_loader, self.validation_loader = \
            self._create_data_loaders(train_validation_split, batch_size, workers, drop_last)
        self.trainer_engine = self._create_trainer_engine()
        self.evaluator_engine = self._create_evaluator_engine()
        self.timer = self._create_timer()
        self.progressbar = tqdm(
            initial=0, leave=False, total=len(self.train_loader), desc=self.progressbar_description.format(0))
        self.writer = SummaryWriter(self.log_directory)
        self._add_event_handlers()
        self._add_model_checkpoint_savers()
        super().__init__(*args, **kwargs)

    """ Abstract properties. """

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
    @abstractmethod
    def trainer_id(self):
        """
        Getter for the trainer id, a unique str to identify the trainer. The corresponding `data` directory sub-folders
        will get a name containing this id.
        :return: the trainer id
        :type: str
        """
        pass

    """ Properties. """

    @property
    def progressbar_description(self):
        """
        The format string to be displayed beside the `tqdm` progressbar.
        :return: the progressbar description string
        :type: str
        """
        return 'TRAINING => loss: {:.6f}'

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

    """ Abstract methods. """

    @abstractmethod
    def _create_evaluator_engine(self):
        """
        Creates an Ignite evaluator engine for the target model.
        :return: an evaluator engine for the target model
        :type: ignite.Engine
        """
        pass

    @abstractmethod
    def _create_trainer_engine(self):
        """
        Creates an Ignite training engine for the target model.
        :return: a trainer engine for the target model
        :type: ignite.Engine
        """
        pass

    """ Properties. """

    @property
    def checkpoint_saver_best(self):
        """
        Checkpoint handler that can be used to save the best performing models
        :return: the best performing checkpoint saver
        :type: ModelCheckpoint
        """
        return ModelCheckpoint(self.checkpoint_directory, filename_prefix='net_best', n_saved=5, atomic=True,
                               create_dir=True, save_as_state_dict=True, require_empty=False)

    @property
    def checkpoint_saver_periodic(self):
        """
        Checkpoint handler that can be used to periodically save model objects to disc.
        :return: the periodic checkpoint saver
        :type: ModelCheckpoint
        """
        return ModelCheckpoint(self.checkpoint_directory, filename_prefix='net_latest', n_saved=3, atomic=True,
                               create_dir=True, save_as_state_dict=True, require_empty=False)

    @property
    def collate_function(self):
        """
        Merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a
        map-style dataset.
        :return: the collate function
        :type: callable
        """
        return None

    """ Event handler methods. """

    def _event_cleanup(self, _):
        """
        Safely cleanup the elements used during training: close the progressbar and log writer.
        """
        self.writer.close()
        self.progressbar.close()

    def _event_log_training_output(self, trainer):
        """
        Write the trainer state output to the progressbar and tensorboard log writer.
        :param trainer: the ignite trainer engine this event was bound to.
        :type: ignite.engine.Engine
        """
        self.writer.add_scalar('training_output', trainer.state.output, self.step)
        self.progressbar.desc = self.progressbar_description.format(trainer.state.output)

    def _event_log_training_results(self, _):
        """
        Run the evaluator engine on the training data and output the results.
        """
        self.evaluator_engine.run(self.train_loader)
        metrics = self.evaluator_engine.state.metrics
        print('\nTraining results - epoch: {}'.format(self.epoch))
        for key, value in metrics.items():
            self.writer.add_scalar('training_{}'.format(key), value, self.step)
            print('{}: {:.6f}'.format(key, value))

    def _event_log_validation_results(self, _):
        """
        Run the evaluator engine on the training data and output the results.
        """
        self.evaluator_engine.run(self.validation_loader)
        metrics = self.evaluator_engine.state.metrics
        print('\nValidation results - epoch: {}'.format(self.epoch))
        for key, value in metrics.items():
            self.writer.add_scalar('validation_{}'.format(key), value, self.step)
            print('{}: {:.6f}'.format(key, value))

    def _event_reset_progressbar(self, _):
        """
        Reset the progressbar in order to be used to track the progress of the next epoch.
        """
        self.progressbar.n = self.progressbar.last_print_n = 0
        self.progressbar.reset(total=len(self.train_loader))

    def _event_update_epoch_counter(self, _):
        """
        Update the AbstractTrainer's epoch counter.
        """
        self.epoch += 1

    def _event_update_progressbar_step(self, _):
        """
        Update the progressbar's counter.
        """
        self.progressbar.update(1)

    def _event_update_step_counter(self, _):
        """
        Update the AbstractTrainer's step counter.
        """
        self.step += 1

    def _event_save_trainer_checkpoint(self, _):
        """
        Create the serialized checkpoint dictionary for the current trainer state, and save it.
        """
        torch.save(self.trainer_checkpoint, os.path.join(self.checkpoint_directory, 'trainer.pth'))

    """ Methods. """

    def _add_model_checkpoint_savers(self):
        """
        Add event handlers for saving model checkpoints.
        """
        checkpoint_saver_best = self.checkpoint_saver_best
        checkpoint_saver_periodic = self.checkpoint_saver_periodic
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver_best, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver_periodic, {'train': self.model})
        self.trainer_engine.add_event_handler(Events.COMPLETED, checkpoint_saver_periodic, {'complete': self.model})

    def _add_event_handlers(self):
        """
        Add event handlers to output common messages and update the progressbar.
        """
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, self._event_log_training_output)
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, self._event_update_progressbar_step)
        self.trainer_engine.add_event_handler(Events.ITERATION_COMPLETED, self._event_update_step_counter)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._event_log_training_results)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._event_log_validation_results)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._event_save_trainer_checkpoint)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._event_reset_progressbar)
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self._event_update_epoch_counter)
        self.trainer_engine.add_event_handler(Events.COMPLETED, self._event_cleanup)

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
        loaders = [DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=drop_last,
                              collate_fn=self.collate_function)
                   for subset in dataset_split_successive(self.dataset, train_validation_split)]
        if not len(loaders[-1]):
            raise ValueError('Empty validation loader. This might be caused by having `drop_last=True` and \
                             a resulting validation set smaller than `batch_size`.')
        return loaders

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
        previous_trainer_checkpoint = torch.load(os.path.join(previous_checkpoint_directory, 'trainer.pth'))
        self.start_epoch = previous_trainer_checkpoint['start_epoch'] + previous_trainer_checkpoint['epochs']

    def run(self):
        """
        Run the trainer.
        """
        self.trainer_engine.run(self.train_loader, max_epochs=self.epochs)
