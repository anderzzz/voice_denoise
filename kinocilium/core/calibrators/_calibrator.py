'''Parent to different training types on the dataset.

The `_Calibrator` class should be inherited by any specific model calibrator.

Written by: Anders Ohrn, March 2022

'''
import sys
import abc

from numpy.random import default_rng

import torch
from torch import optim

class CalibratorInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Calibrator subclasses. Any class inheriting `_Calibrator` will have to satisfy this
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

class _Calibrator(CalibratorInterface):
    '''Parent class

    Args:

    '''
    def __init__(self,
                 criterion,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler=None,
                 device=None,
                 run_label='Calibrator Run Label',
                 random_seed=None,
                 f_out=sys.stdout,
                 save_tmp_name='model_in_training',
                 num_workers=0,
                 deterministic=True):

        self.criterion = criterion
        self.optimizer_label = optimizer_label
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.run_label = run_label
        self.random_seed = random_seed
        self.f_out = f_out
        self.save_tmp_name = save_tmp_name
        self.num_workers = num_workers
        self.deterministic = deterministic

        self.optimizer = None

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.rng = default_rng(self.inp_random_seed)
        torch.manual_seed(self.rng.integers(2**63))
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def connect_calibration_params(self, parameters):
        '''Bla bla

        '''
        self._optimizer_method = getattr(optim, self.optimizer_label)
        self.optimizer = self._optimizer_method(parameters=parameters, **self.optimizer_kwargs)

#    def train(self, n_epochs):
#        self.model.train()
#        for k_epoch in range(n_epochs):
#            for data_inputs in self.dataloaders['train']:
#                self.model(data_inputs)

def _calibrate_params_(loss, optimizer, lr_scheduler=None):
    loss.backward()
    optimizer.step()
    if not lr_scheduler is None:
        lr_scheduler.step()

def _generate_optimizer(label, parameters, kwargs={}):
    try:
        opt_method = getattr(optim, label)
    except AttributeError:
        raise AttributeError('The optimizer label {} is not a method part of `torch.optim`.'.format(label))
    try:
        optimizer = opt_method(parameters, **kwargs)
    except TypeError:
        raise AttributeError('Invalid optimizer keyword argument obtained')
    return optimizer

def _generate_lr_scheduler(label, optimizer, kwargs={}):
    try:
        lr_scheduler_method = getattr(optim.lr_scheduler, label)
    except:
        raise AttributeError('The learning-rate scheduler label {} is not a method part of `torch.optim.lr_scheduler`.')
    try:
        lr_scheduler = lr_scheduler_method(optimizer, **kwargs)
    except TypeError:
        raise AttributeError('Invalid learning-rate scheduler keyword argument obtained')
    return lr_scheduler

class _CalibratorPairedTensors(object):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters, optimizer_label, optimizer_kwargs,
                 lr_scheduler_label, lr_scheduler_kwargs):
        self.optimizer = _generate_optimizer(optimizer_label, optimizer_parameters, optimizer_kwargs)
        if lr_scheduler_label is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = _generate_lr_scheduler(lr_scheduler_label, self.optimizer, lr_scheduler_kwargs)

    def train(self, model, n_epochs, dataloader):
        '''Bla bla

        '''
        model.train()
        for epoch in range(n_epochs):
            self.reporter.reset()
            for data_inputs in dataloader:
                self.optimizer.zero_grad()
                loss, _ = self.cmp_prediction_loss(model, data_inputs)
                _calibrate_params_(loss, self.optimizer, self.lr_scheduler)
                self.reporter.append(loss.item(), epoch)
            self.reporter.report()

    def cmp_prediction_loss(self, model, data_inputs):
        raise NotImplementedError('The method `cmp_predict_loss` should be implemented in child class')

    def cmp_batch_size(self, data_inputs):
        raise NotImplementedError('The method `cmp_batch_size` should be implemented in child class')

class _CalibratorClassLabel(CalibratorInterface):
    '''Bla bla

    '''
    def __init__(self):
        pass


    def train(self, model, n_epochs, dloader):
        '''Bla bla

        '''
        model.train()
        for epoch in range(n_epochs):

            running_loss = 0.0
            running_correct = 0
            for data_inputs in dloader:
                self.optimizer.zero_grad()
                loss, prediction = self._cmp_prediction_loss(data_inputs)
                _calibrate_params_(loss, self.optimizer, self.lr_scheduler)
                running_loss += loss.item() * size_batch
                running_correct += torch.sum(prediction == XXX)



def progress_bar(current, total, barlength=20):
    '''Print progress of training of a batch. Helpful in PyCharm'''
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')