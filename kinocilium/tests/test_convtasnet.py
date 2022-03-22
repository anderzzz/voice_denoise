'''Bla bla

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data1'

# CHECK constants. The two files have number of frames 325680 and 190944. So the chunker returns 325680 // (SR * N_SECS)
# number of chunks. So 5+2=7, which means if batch size of 4, we get two iterations from the DataLoader of batch size
# 4 and 3.
SR = 16000
N_SECS = 4
BATCH_SIZE = 4
CHECK_DIMS_IN = iter([(4, 1, SR * N_SECS), (3, 1, SR * N_SECS)])
CHECK_DIMS_OUT = iter([(4, 2, SR * N_SECS), (3, 2, SR * N_SECS)])

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader
import torch
torch.manual_seed(42)

from kinocilium.core.models import factory as factory_model
from kinocilium.core.data_getters import factory as factory_data

def test_simple_init_and_call():
    convtas_net = factory_model.create('conv_tasnet')
    plain_audio = factory_data.create('plain wav', path_to_folder=abs_data_path, read_metadata=True, slice_size=SR * N_SECS)
    plain_audio_loader = DataLoader(plain_audio, batch_size=BATCH_SIZE, shuffle=False)

    convtas_net.eval()
    for dd in plain_audio_loader:
        xx = convtas_net(dd['waveform'])

        assert tuple(dd['waveform'].shape) == CHECK_DIMS_IN.__next__()
        assert tuple(xx.shape) == CHECK_DIMS_OUT.__next__()


if __name__ == '__main__':
    test_simple_init_and_call()