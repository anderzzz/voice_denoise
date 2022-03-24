'''Calibrator modules for various types of audio tasks

Written by: Anders Ohrn, March 2022

'''
import sys
from torch import optim

from kinocilium.core.calibrators._calibrator import _CalibratorPairedTensors


class CalibratorPairedAudio(_CalibratorPairedTensors):
    '''Bla bla

    '''
    def __init__(self,
                 optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 run_label='Calibrator Run Label',
                 random_seed=None,
                 f_out=sys.stdout,
                 save_tmp_name='model_in_training',
                 num_workers=0,
                 deterministic=True):

        super(CalibratorPairedAudio, self).__init__(
            optimizer_parameters=optimizer_parameters,
            optimizer_label=optimizer_label,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_label=lr_scheduler_label,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )

    def cmp_prediction_loss(self, model, data_inputs):
        '''Compute the prediction given input and model, as well as the loss

        '''
        denoised_ = model(data_inputs['waveform_noisy_speech'])
        loss = self.criterion(denoised_, data_inputs['waveform_clean_speech'])

        return loss, denoised_

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def validate(self, model, dataloader_validate):
        '''Bla bla

        '''
        model.eval()
        for data_inputs in dataloader_validate:
            audio_noisy = data_inputs
            audio_clean = data_inputs

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


class CalibratorPairedAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, optimizer_parameters,
                 optimizer_label,
                 optimizer_kwargs,
                 lr_scheduler_label,
                 lr_scheduler_kwargs,
                 run_label='Calibrator Run Label',
                 random_seed=None,
                 f_out=sys.stdout,
                 save_tmp_name='model_in_training',
                 num_workers=0,
                 deterministic=True):
        self._instance = CalibratorPairedAudio(optimizer_parameters,
                                               optimizer_label,
                                               optimizer_kwargs,
                                               lr_scheduler_label,
                                               lr_scheduler_kwargs,
                                               run_label,
                                               random_seed,
                                               f_out,
                                               save_tmp_name,
                                               num_workers,
                                               deterministic)
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance