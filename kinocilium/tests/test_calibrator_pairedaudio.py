'''Bla bla

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data2'

SR = 16000
N_SECS = 4
BATCH_SIZE = 1

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader
import torch
torch.manual_seed(42)

from kinocilium.core.models import factory as factory_model
from kinocilium.core.data_getters import factory as factory_data
from kinocilium.core.calibrators import factory as factory_calibrator
from kinocilium.core.reporter import ReporterRunningLoss

def test_simple_init_and_train():
    convtas_net = factory_model.create('conv_tasnet')
    paired_audio = factory_data.create('ms-snsd',
                               path_to_noisyspeech=abs_data_path,
                               path_to_cleanspeech=abs_data_path,
                               path_to_noise=abs_data_path,
                               read_metadata=False)

    reporter = ReporterRunningLoss(dataset_size=len(paired_audio),
                                   dataloader_validate=DataLoader(paired_audio))
    calibrator_paired = factory_calibrator.create('paired audio recreation',
                                                  optimizer_parameters=convtas_net.parameters(),
                                                  optimizer_label='SGD',
                                                  optimizer_kwargs={'lr':0.001},
                                                  lr_scheduler_label='StepLR',
                                                  lr_scheduler_kwargs={'step_size':10},
                                                  reporter=reporter)

    calibrator_paired.train(model=convtas_net,
                            n_epochs=2,
                            dataloader=DataLoader(paired_audio, batch_size=BATCH_SIZE, shuffle=False))

if __name__ == '__main__':
    test_simple_init_and_train()