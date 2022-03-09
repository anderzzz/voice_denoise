'''Audio Dataset classes

Written By: Anders Ohrn, March 2022

'''
from pathlib import Path
import re

import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset

class _AudioUnlabelledDataset(Dataset):
    '''Parent class for unlabelled audio data of the PyTorch Dataset format

    '''
    def __init__(self, file_path_getter, len_file_paths, read_metadata):
        super(_AudioUnlabelledDataset, self).__init__()
        self.file_path_getter = file_path_getter
        self.len_file_paths = len_file_paths
        self.read_metadata = read_metadata

    def __len__(self):
        return self.len_file_paths

    def __getitem__(self, idx):
        '''Retrieve raw data from disk

        Args:
            idx: index to retrieve

        Returns:
            raw_data (dict)

        '''
        path = self.file_path_getter(idx)
        if self.read_metadata:
            metadata = torchaudio.info(path)
        else:
            metadata = None

        waveform, sample_rate = torchaudio.load(path)

        return {'waveform' : waveform, 'sample_rate' : sample_rate, 'metadata' : metadata}

class _AudioLabelledDataset(Dataset):
    '''Parent class for labelled audio data of the PyTorch Dataset format

    '''
    def __init__(self):
        super(_AudioLabelledDataset, self).__init__()

class _AudioPairedDataset(Dataset):
    '''Parent class for paired audio data of the Pytorch Dataset format

    '''
    def __init__(self, file_paths_getter, len_files_path, keys_filetype, read_metadata):
        super(_AudioPairedDataset, self).__init__()
        self.file_path_getter = file_paths_getter
        self.len_files_path = len_files_path
        self.keys_filetype = keys_filetype
        self.read_metadata = read_metadata

    def __len__(self):
        return self.len_files_path

    def __getitem__(self, idx):
        '''Bla bla

        '''
        path_file, path_file_counterpart = self.file_path_getter(idx)
        if self.read_metadata:
            metadata = torchaudio.info(path_file)
            metadata_counterpart = torchaudio.info(path_file_counterpart)

        waveform, sample_rate = torchaudio.load(path_file)
        waveform_counterpart, sample_rate_counterpart = torchaudio.load(path_file_counterpart)
        print (waveform)
        print (waveform_counterpart)
        raise RuntimeError

        return None

class NoisySpeechLabelledData(_AudioLabelledDataset):
    '''Bla bla

    '''
    def __init__(self, generator_type):
        super(NoisySpeechLabelledData, self).__init__()

        self.generator_type = generator_type

        if self.generator_type == 'MS-SNSD':
            self._generator = self._generator_ms_snsd

        else:
            raise ValueError('The generator type {} is undefined.'.format(self.generator_type))

class AudioPlainWAVData(_AudioUnlabelledDataset):
    '''Bla bla

    '''
    def __init__(self, path_to_folder, filter_on_suffix,
                 read_metadata=True):
        p = Path(path_to_folder)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(path_to_folder))

        if filter_on_suffix is None:
            filter_condition = '*'
        else:
            filter_condition = '*.{}'.format(filter_on_suffix)
        file_paths = list(p.glob(filter_condition))
        n_file_paths = len(file_paths)

        super(AudioPlainWAVData, self).__init__(file_path_getter=lambda idx: file_paths[idx],
                                                len_file_paths=n_file_paths,
                                                read_metadata=read_metadata)

class MSSNSDNoisySpeechData(_AudioPairedDataset):
    '''Bla bla

    '''
    CLEAN_FNAME_ROOT = 'clnsp'
    NOISE_FNAME_ROOT = 'noisy'
    SPEECH2NOISE_ROOT = 'SNRdb'
    FILE_SUFFIX = 'wav'

    def __init__(self, path_to_noisyspeech, path_to_cleanspeech, path_to_noise,
                 return_clean_counterpart=True, filter_on_snr='[0-9]+\.[0-9]+',
                 read_metadata=True):
        self.p_noisyspeech = Path(path_to_noisyspeech)
        if not self.p_noisyspeech.is_dir():
            raise ValueError('File folder {} not found'.format(self.path_to_noisyspeech))
        self.p_cleanspeech = Path(path_to_cleanspeech)
        if not self.p_cleanspeech.is_dir():
            raise ValueError('File folder {} not found'.format(self.path_to_cleanspeech))
        self.p_noise = Path(path_to_noise)
        if not self.p_noise.is_dir():
            raise ValueError('File folder {} not found'.format(self.path_to_noise))

        self.return_clean_counterpart = return_clean_counterpart
        if self.return_clean_counterpart:
            ret_keys = ('noisy_speech', 'clean_speech')
        else:
            ret_keys = ('noisy_speech', 'noise')

        re_expr = '{}[0-9]+_{}_{}_{}[0-9]+.{}'.format(self.NOISE_FNAME_ROOT, self.SPEECH2NOISE_ROOT,
                                                      filter_on_snr,
                                                      self.CLEAN_FNAME_ROOT, self.FILE_SUFFIX)
        re_expr_ = re.compile(re_expr)
        self._file_paths = [s for s in self.p_noisyspeech.glob('*') if re_expr_.match(s.name)]
        n_file_paths = len(self._file_paths)

        super(MSSNSDNoisySpeechData, self).__init__(file_paths_getter=self._p_getter,
                                                    len_files_path=n_file_paths,
                                                    keys_filetype=ret_keys,
                                                    read_metadata=read_metadata)

    def _p_getter(self, idx):
        '''Bla bla

        '''
        p_file_noisyspeech = Path(self._file_paths[idx])
        all_numbers = re.findall('[0-9]+\.?\d+', p_file_noisyspeech.name)
        if len(all_numbers) != 3:
            raise RuntimeError('Encountered file without three numbers: {}'.format(p_file_noisyspeech.name))

        noisy_file_name = '{}{}_{}_{}.{}'.format(self.NOISE_FNAME_ROOT,
                                                 all_numbers[0],
                                                 self.SPEECH2NOISE_ROOT,
                                                 all_numbers[1],
                                                 self.FILE_SUFFIX)
        clean_file_name = '{}{}.{}'.format(self.CLEAN_FNAME_ROOT,
                                           all_numbers[2],
                                           self.FILE_SUFFIX)

        print (p_file_noisyspeech)
        print (self.p_cleanspeech.joinpath(clean_file_name))

        if self.return_clean_counterpart:
            ret_counterpart = self.p_cleanspeech.joinpath(clean_file_name)
        else:
            ret_counterpart = self.p_noisyspeech.joinpath(noisy_file_name)

        print (p_file_noisyspeech, ret_counterpart)

        return (p_file_noisyspeech, ret_counterpart)

    def _match_clean(self, name):
        pass


class AudioMSSNSDDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_noisyspeech, path_to_cleanspeech, path_to_noise,
                 read_metadata=True):
        self._instance = MSSNSDNoisySpeechData(path_to_noisyspeech=path_to_noisyspeech,
                                               path_to_cleanspeech=path_to_cleanspeech,
                                               path_to_noise=path_to_noise,
                                               read_metadata=True)

        return self._instance

class AudioPlainWAVDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_folder, filter_on_suffix='wav', read_metadata=True):
        self._instance = AudioPlainWAVData(path_to_folder=path_to_folder,
                                           filter_on_suffix=filter_on_suffix,
                                           read_metadata=read_metadata)

        return self._instance

class AudioDataFactory(object):
    '''Interface to audio data factories.

    Typical usage involves the invocation of the `create` method, which returns a specific audio dataset

    '''
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: A Fungi Data Builder instance

        '''
        self._builders[key] = builder

    @property
    def keys(self):
        return self._builders.keys()

    def create(self, key, **kwargs):
        '''Method to create audio data set through uniform interface

        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered data builder: {}'.format(key))
        return builder(**kwargs)

factory = AudioDataFactory()
factory.register_builder('plain wav', AudioPlainWAVDataBuilder())
factory.register_builder('ms-snsd', AudioMSSNSDDataBuilder())

