'''Unit tests of audiodata classes

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data1'

# Constants: check values to test against
CHECK_KEYS = {'waveform', 'sample_rate', 'metadata'}

import pytest

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from kinocilium.core.audiodata import factory

data = factory.create('plain wav', path_to_folder=abs_data_path, read_metadata=False)
for dd in data:
    assert set(dd.keys()) == CHECK_KEYS
