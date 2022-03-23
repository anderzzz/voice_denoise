'''Parent to different training types on the dataset.

The `_Learner` class should be inherited by any specific model learner.

Written by: Anders Ohrn, March 2022

'''
import sys
import abc

from numpy.random import default_rng

import torch
from torch import optim

class LearnerInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Learner subclasses. Any class inheriting `_Learner` will have to satisfy this
    interface, otherwise it will not instantiate
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'validate') and
                callable(subclass.eval) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model))

    @abc.abstractmethod
    def train(self, n_epochs: int):
        '''Train model'''
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self, **kwargs):
        '''Evaluate model'''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError

class _Learner(LearnerInterface):
    '''Parent class

    Args:

    '''
    def __init__(self, run_label='Learner Run Label',
                       random_seed=None,
                       f_out=sys.stdout,
                       save_tmp_name='model_in_training',
                       num_workers=0,
                       deterministic=True):

        self.run_label = run_label
        self.random_seed = random_seed
        self.f_out = f_out
        self.save_tmp_name = save_tmp_name
        self.num_workers = num_workers
        self.deterministic = deterministic

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.rng = default_rng(self.inp_random_seed)
        torch.manual_seed(self.rng.integers(2**63))
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            shuffle_me = not self.deterministic
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            shuffle_me = self.deterministic

        self.optimizer = None
        self.lr_scheduler = None
        self.dataloaders = None
        self.model = None
        self.progress_reporter = None

    def set_sgd_optim(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0,
                      scheduler_step_size=15, scheduler_gamma=0.1):
        '''Override the `optimizer` and `lr_scheduler` attributes with an SGD optimizer and an exponential decay
        learning rate.
        This is a convenience method for a common special case of the optimization. A child class can define other
        PyTorch optimizers and learning-rate decay methods.
        Args:
            parameters: The parameters of the model to optimize
            lr (float, optional): Initial learning rate. Defaults to 0.01
            momentum (float, optional): Momentum of SGD. Defaults to 0.9
            weight_decay (float, optional): L2 regularization of weights. Defaults to 0.0 (no weight regularization)
            scheduler_step_size (int, optional): Steps between learning-rate update. Defaults to 15
            scheduler_gamma (float, optional): Factor to reduce learning-rate with. Defaults to 0.1.
        '''
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                      step_size=scheduler_step_size,
                                                      gamma=scheduler_gamma)

def progress_bar(current, total, barlength=20):
    '''Print progress of training of a batch. Helpful in PyCharm'''
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')