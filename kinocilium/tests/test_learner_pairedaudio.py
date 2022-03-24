'''Bla bla

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data2'

SR = 16000
N_SECS = 4
BATCH_SIZE = 4

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader
import torch
torch.manual_seed(42)

from kinocilium.core.models import factory as factory_model
from kinocilium.core.data_getters import factory as factory_data
from kinocilium.core.learners import factory as factory_learner

def test_simple_init_and_train():
    convtas_net = factory_model.create('conv_tasnet')
    plain_audio = factory_data.create('plain wav', path_to_folder=abs_data_path, read_metadata=True, slice_size=SR * N_SECS)
    learner_paired = factory_learner.create('paired audio recreation')

    learner_paired.model = convtas_net
    learner_paired.dataloaders = {
        'train' : DataLoader(plain_audio, batch_size=BATCH_SIZE, shuffle=False),
        'validate' : DataLoader(plain_audio, batch_size=BATCH_SIZE, shuffle=False),
    }

    learner_paired.train(model=convtas_net, n_epochs=XXX).with_data_().with_reporter_()
