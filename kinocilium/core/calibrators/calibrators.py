'''Calibrator modules for various types of audio tasks

Written by: Anders Ohrn, March 2022

'''
import sys
import torch

from kinocilium.core.calibrators._calibrator import _Calibrator

class CalibratorLabelledAudio(_Calibrator):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 reporter,
                 model_save_path,
                 model_load_path,
                 device=None,
                 random_seed=None,
                 num_workers=0,
                 deterministic=True):

        super(CalibratorLabelledAudio, self).__init__(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            reporter=reporter,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

        self.criterion = torch.nn.CrossEntropyLoss()

    def cmp_prediction_loss(self, model, data_inputs):
        x_inp = data_inputs[1]['waveform']
        y_label = data_inputs[0]
        y_label_predict = model(x_inp)
        loss = self.criterion(y_label_predict, y_label)
        print (y_label, y_label_predict)
        return loss, y_label_predict

    def cmp_batch_size(self, data_inputs):
        return data_inputs[1]['waveform'].size(0)

    def save_model_state(self, model):
        '''Save model state to disk

        '''
        torch.save({'Model State by {}'.format(self.__class__.__name__) : model.state_dict()},
                   '{}.tar'.format(self.model_save_path))

    def load_model_state(self, model):
        '''Load model state from disk

        '''
        saved_dict = torch.load('{}.tar'.format(self.model_load_path))
        return model.load_state_dict(saved_dict['Model State by {}'.format(self.__class__.__name__)])


class CalibratorPairedAudio(_Calibrator):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 reporter,
                 model_save_path,
                 model_load_path,
                 device=None,
                 random_seed=None,
                 num_workers=0,
                 deterministic=True):

        super(CalibratorPairedAudio, self).__init__(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            reporter=reporter,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

    def cmp_prediction_loss(self, model, data_inputs):
        '''Compute the prediction given input and model, as well as the loss

        '''
        denoised_ = model(data_inputs['waveform_noisy_speech'])
        loss = self.criterion(denoised_, data_inputs['waveform_clean_speech'])

        return loss, denoised_

    def cmp_batch_size(self, data_inputs):
        '''Compute batch size implied by the given data tensor

        '''
        return data_inputs['waveform_noisy_speech'].size(0)

    def save_model_state(self, model):
        '''Save model state to disk

        '''
        torch.save({'Model State by {}'.format(self.__class__.__name__) : model.state_dict()},
                   '{}.tar'.format(self.model_save_path))

    def load_model_state(self, model):
        '''Load model state from disk

        '''
        saved_dict = torch.load('{}.tar'.format(self.model_load_path))
        return model.load_state_dict(saved_dict['Model State by {}'.format(self.__class__.__name__)])


class CalibratorPairedAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 reporter,
                 run_label='Calibrator Run Label',
                 random_seed=None,
                 f_out=sys.stdout,
                 save_tmp_name='model_in_training',
                 num_workers=0,
                 deterministic=True):
        self._instance = CalibratorPairedAudio(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            reporter=reporter
        )
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance

class CalibratorLabelledAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, optimizer_parameters,
                 optimizer_label='SGD',
                 optimizer_kwargs={'lr':0.01},
                 lr_scheduler_label=None,
                 lr_scheduler_kwargs={},
                 reporter=None,
                 model_save_path='./dummy',
                 model_load_path=None,
                 run_label='Calibrator Run Label',
                 random_seed=None,
                 f_out=sys.stdout,
                 save_tmp_name='model_in_training',
                 num_workers=0,
                 deterministic=True):
        self._instance = CalibratorLabelledAudio(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            reporter=reporter,
            model_save_path=model_save_path,
            model_load_path=model_load_path
        )
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance
