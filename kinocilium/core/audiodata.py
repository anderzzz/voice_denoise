'''Audio Dataset classes

Written By: Anders Ohrn, March 2022

'''
from pathlib import Path

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

class AudioMSSNSDDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self):
        self._instance = NoisySpeechLabelledData(None)

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

#
# TESTS
#
