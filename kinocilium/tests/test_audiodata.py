'''Unit tests of audiodata classes

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data1'

# Constants: check values to test against
CHECK_KEYS_0 = {'waveform', 'sample_rate', 'metadata'}
CHECK_KEYS_1 = {'waveform', 'sample_rate'}
CHECK_METADATA_KEYS = {'num_frames', 'bits_per_sample', 'num_channels', 'encoding', 'sample_rate'}
CHECK_NUM_FRAMES = {'AirConditioner_6' : 325680, 'NeighborSpeaking_13' : 190944}

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader

from kinocilium.core.data_getters import factory

def test_simple_reads():
    data = factory.create('plain wav', path_to_folder=abs_data_path, read_metadata=True)
    for dd in data:
        assert set(dd.keys()) == CHECK_KEYS_0

    data = factory.create('plain wav', path_to_folder=abs_data_path, read_metadata=False)
    for dd in data:
        assert set(dd.keys()) == CHECK_KEYS_1

def test_metadata_collate():
    data = factory.create('plain wav', path_to_folder=abs_data_path, read_metadata=True)
    dloader = DataLoader(dataset=data, batch_size=1)
    for dd in dloader:
        assert set(dd['metadata'].keys()) == CHECK_METADATA_KEYS

def test_simple_chunked_reads():
    SR = 16000
    N_SECS = 5
    data = factory.create('plain wav', path_to_folder=abs_data_path, read_metadata=True, slice_size=SR * N_SECS)
    assert len(data) == sum([v // (SR * N_SECS) for v in CHECK_NUM_FRAMES.values()])

    n_len = 0
    for dd in data:
        assert dd['metadata']['num_frames'] == SR * N_SECS
        assert dd['metadata']['sample_rate'] == SR
        n_len += 1
    assert n_len == len(data)

if __name__ == '__main__':
    test_simple_reads()
    test_metadata_collate()
    test_simple_chunked_reads()