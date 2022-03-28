'''Parent to different training types on the dataset.

The `_Calibrator` class should be inherited by any specific model calibrator.

Written by: Anders Ohrn, March 2022

'''
import sys
import abc

from numpy.random import default_rng

import torch
from torch import optim

from kinocilium.core.reporter import ReporterNull

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
    def train(self, model, n_epochs, dataloader, dataloader_validate):
        '''Train model'''
        raise NotImplementedError

    @abc.abstractmethod
    def cmp_prediction_loss(self, model, data_inputs):
        '''Compute the loss and prediction given model and mini-batch of data'''
        raise NotImplementedError

    @abc.abstractmethod
    def cmp_batch_size(self, data_inputs):
        '''Compute the batch size given mini-batch of data'''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model_state(self, model):
        '''Save model state to file'''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model_state(self, model):
        '''Load model state from file'''
        raise NotImplementedError

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

class _Calibrator(CalibratorInterface):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 lr_scheduler_label,
                 reporter=None,
                 model_save_path=None,
                 model_load_path=None,
                 device=None,
                 random_seed=42,
                 deterministic=True,
                 optimizer_kwargs={}, lr_scheduler_kwargs={}
                 ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.random_seed = random_seed
        self.rng = default_rng(self.random_seed)
        torch.manual_seed(self.rng.integers(2**63))
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        self.optimizer = _generate_optimizer(optimizer_label, optimizer_parameters, optimizer_kwargs)
        if lr_scheduler_label is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = _generate_lr_scheduler(lr_scheduler_label, self.optimizer, lr_scheduler_kwargs)

        if reporter is None:
            self.reporter = ReporterNull()
        else:
           self.reporter = reporter
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

    def train(self, model, n_epochs, dataloader, dataloader_validate=None):
        '''Bla bla

        '''
        for epoch in range(n_epochs):
            model.train()
            self.reporter.reset()
            for k_batch, data_inputs in enumerate(dataloader):
                self.optimizer.zero_grad()
                loss, prediction = self.cmp_prediction_loss(model, data_inputs)
                loss.backward()

                self.optimizer.step()
                if not self.lr_scheduler is None:
                    self.lr_scheduler.step()

                self.reporter.append(
                    descriptor='Training Loop Report Vector',
                    loss=loss.item(),
                    prediction=prediction,
                    data_inputs=data_inputs,
                    epoch=epoch,
                    k_batch=k_batch,
                    minibatch_size=self.cmp_batch_size(data_inputs)
                )
            self.reporter.report()

            if not dataloader_validate is None:
                model.eval()
                self.reporter.reset()
                for k_batch, data_inputs in enumerate(dataloader_validate):
                    loss, prediction = self.cmp_prediction_loss(model, data_inputs)

                    self.reporter.append(
                        descriptor='Validation Loop Report Vector',
                        loss=loss.item(),
                        prediction=prediction,
                        data_inputs=data_inputs,
                        epoch=epoch,
                        k_batch=k_batch,
                        minibatch_size=self.cmp_batch_size(data_inputs)
                    )
                self.reporter.report()

            self.save_model_state(model)

    def cmp_prediction_loss(self, model, data_inputs):
        raise NotImplementedError('The method `cmp_predict_loss` should be implemented in child class')

    def cmp_batch_size(self, data_inputs):
        raise NotImplementedError('The method `cmp_batch_size` should be implemented in child class')

    def save_model_state(self, model):
        '''Save model state to disk. This can be overridden in child classes if needed.

        '''
        torch.save({'Model State by {}'.format(self.__class__.__name__) : model.state_dict()},
                   '{}.tar'.format(self.model_save_path))

    def load_model_state(self, model):
        '''Load model state from disk. This can be overridden in child classes if needed.

        '''
        saved_dict = torch.load('{}.tar'.format(self.model_load_path))
        model.load_state_dict(saved_dict['Model State by {}'.format(self.__class__.__name__)])
        return model

def progress_bar(current, total, barlength=20):
    '''Print progress of training of a batch. Helpful in PyCharm'''
    percent = float(current) / total
    arrow = '-' * int(percent * barlength - 1) + '>'
    spaces = ' ' * (barlength - len(arrow))
    print ('\rProgress: [{}{}]'.format(arrow, spaces), end='')