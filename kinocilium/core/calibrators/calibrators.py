'''Calibrator modules for various types of audio tasks

Written by: Anders Ohrn, March 2022

'''
import torch

from kinocilium.core.calibrators._calibrator import _Calibrator
from kinocilium.core.calibrators.reporter import ReporterClassification
from kinocilium.core.loss_criteria import ScaleInvariantSource2DistortionRatio, Source2DistortionRatio

class CalibratorLabelledAudio(_Calibrator):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 model_save_path=None,
                 model_load_path=None,
                 device=None,
                 random_seed=None,
                 deterministic=True):

        super(CalibratorLabelledAudio, self).__init__(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_save_path=model_save_path,
            model_load_path=model_load_path,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def cmp_prediction_loss(self, model, data_inputs):
        x_inp = data_inputs[1]['waveform'].to(self.device)
        y_label = data_inputs[0]
        _y_label_predict = model(x_inp)
        loss = self.criterion(_y_label_predict, y_label)
        _, y_label_predict = torch.max(_y_label_predict, 1)
        return loss, y_label_predict

    def cmp_batch_size(self, data_inputs):
        return data_inputs[1]['waveform'].size(0)

    def train(self, model, n_epochs, dataloader, dataloader_validate=None, reporter=None):
        if reporter is None:
            reporter_ = ReporterClassification()
        else:
            reporter_ = reporter
        super().train(model=model, n_epochs=n_epochs, dataloader=dataloader,
                      dataloader_validate=dataloader_validate, reporter=reporter_)

class CalibratorPairedAudio(_Calibrator):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 model_save_path,
                 model_load_path,
                 loss_type,
                 device=None,
                 random_seed=None,
                 deterministic=True):

        super(CalibratorPairedAudio, self).__init__(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_save_path=model_save_path,
            model_load_path=model_load_path,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )

        if isinstance(loss_type, str):
            if loss_type == 'SI-SDR':
                self.criterion = ScaleInvariantSource2DistortionRatio()
            elif loss_type == 'SDR':
                self.criterion = Source2DistortionRatio()
            else:
                raise ValueError('Unknown `loss_type` specification: {}'.format(loss_type))

        elif isinstance(loss_type, torch.nn.Module):
            self.criterion = loss_type

        else:
            raise ValueError('The `loss_type` should be either a string or a PyTorch Module')

    def cmp_prediction_loss(self, model, data_inputs):
        '''Compute the prediction given input and model, as well as the loss

        '''
        denoised_ = model(data_inputs['waveform_noisy_speech'])
        loss = self.criterion(denoised_, data_inputs['waveform_clean_speech'])
        raise RuntimeError()

        return loss, denoised_

    def cmp_batch_size(self, data_inputs):
        '''Compute batch size implied by the given data tensor

        '''
        return data_inputs['waveform_noisy_speech'].size(0)

class CalibratorPairedAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, optimizer_parameters,
                 optimizer_label='SGD',
                 optimizer_kwargs={'lr':0.001},
                 lr_scheduler_label=None,
                 lr_scheduler_kwargs={},
                 model_save_path=None,
                 model_load_path=None,
                 loss_type='SI-SDR',
                 device=None,
                 random_seed=None,
                 deterministic=True):
        self._instance = CalibratorPairedAudio(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_save_path=model_save_path,
            model_load_path=model_load_path,
            loss_type=loss_type,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance

class CalibratorLabelledAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, optimizer_parameters,
                 optimizer_label='SGD',
                 optimizer_kwargs={'lr':0.001},
                 lr_scheduler_label=None,
                 lr_scheduler_kwargs={},
                 model_save_path=None,
                 model_load_path=None,
                 device=None,
                 random_seed=None,
                 deterministic=True):
        self._instance = CalibratorLabelledAudio(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_save_path=model_save_path,
            model_load_path=model_load_path,
            device=device,
            random_seed=random_seed,
            deterministic=deterministic
        )
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance
