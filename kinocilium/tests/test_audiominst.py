'''Unit tests of audiodata classes

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data3'

# Constants: check values to test against
CHECK_LABELS = [0,1,2,3,4,5,6,7,8,9]
CHECK_AUDIO_KEYS = {'waveform', 'sample_rate'}

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from kinocilium.core.data_getters import factory

def test_simple_read():
    data = factory.create('audio-minst', path_to_folder=abs_data_path, label='digit', read_metadata=False, slice_size=None)
    for label, audio_data in data:
        try:
            CHECK_LABELS.remove(label)
        except ValueError:
            raise AssertionError('Returned label either outside range or duplicate: {}'.format(label))

        assert set(audio_data.keys()) == CHECK_AUDIO_KEYS
    assert len(CHECK_LABELS) == 0

if __name__ == '__main__':
    test_simple_read()