'''Unit tests of audiodata classes

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data2'

# Constants: check values to test against
CHECK_KEYS_1 = {'waveform_noisy_speech', 'sample_rate_noisy_speech', 'metadata_noisy_speech',
                'waveform_clean_speech', 'sample_rate_clean_speech', 'metadata_clean_speech'}
CHECK_KEYS_2 = {'waveform_noisy_speech', 'sample_rate_noisy_speech',
                'waveform_noise', 'sample_rate_noise'}
N_FILES_SNR40 = 1
CHECK_NUM_FRAMES_FILES = {'clnsp12' : 170125, 'clnsp3' : 206963}

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from kinocilium.core.data_getters import factory

def test_basic_retrieval():
    data = factory.create('ms-snsd',
                          path_to_noisyspeech=abs_data_path,
                          path_to_cleanspeech=abs_data_path,
                          path_to_noise=abs_data_path,
                          read_metadata=True)
    for dd in data:
        assert set(dd.keys()) == CHECK_KEYS_1

def test_noise_retrieval():
    data = factory.create('ms-snsd',
                          path_to_noisyspeech=abs_data_path,
                          path_to_cleanspeech=abs_data_path,
                          path_to_noise=abs_data_path,
                          return_clean_counterpart=False,
                          read_metadata=False)
    for dd in data:
        assert set(dd.keys()) == CHECK_KEYS_2

def test_select_file_subset():
    data = factory.create('ms-snsd',
                          path_to_noisyspeech=abs_data_path,
                          path_to_cleanspeech=abs_data_path,
                          path_to_noise=abs_data_path,
                          return_clean_counterpart=True,
                          filter_on_snr='40\.0',
                          read_metadata=False)
    assert len(data) == N_FILES_SNR40
    counter = 0
    for dd in data:
        counter += 1
    assert counter == N_FILES_SNR40

def test_basic_chunked_retrieval():
    data = factory.create('ms-snsd',
                          path_to_noisyspeech=abs_data_path,
                          path_to_cleanspeech=abs_data_path,
                          path_to_noise=abs_data_path,
                          read_metadata=True,
                          slice_size=16001)
    assert len(data) == sum([x // 16001 for x in CHECK_NUM_FRAMES_FILES.values()])
    counter = 0
    for dd in data:
        assert dd['metadata_noisy_speech']['num_frames'] == 16001
        assert tuple(dd['waveform_noisy_speech'].shape) == (1, 16001)
        assert dd['metadata_clean_speech']['num_frames'] == 16001
        assert tuple(dd['waveform_clean_speech'].shape) == (1, 16001)
        counter += 1
    assert counter == len(data)

if __name__ == '__main__':
    test_basic_retrieval()
    test_noise_retrieval()
    test_select_file_subset()
    test_basic_chunked_retrieval()