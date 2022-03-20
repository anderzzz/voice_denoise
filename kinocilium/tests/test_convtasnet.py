'''Bla bla

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data1'

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader

from kinocilium.core.models import factory as factory_model
from kinocilium.core.data_getters import factory as factory_data

def test_simple_init():
    convtas_net = factory_model.create('conv_tasnet')
    plain_audio = factory_data.create('plain wav', path_to_folder=abs_data_path, read_metadata=True)
    plain_audio_loader = DataLoader(plain_audio, batch_size=1, shuffle=False)

    convtas_net.eval()
    for dd in plain_audio_loader:
        convtas_net(dd['waveform'])


if __name__ == '__main__':
    test_simple_init()