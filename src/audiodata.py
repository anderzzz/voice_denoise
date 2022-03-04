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
    def __init__(self, file_path_getter):
        super(_AudioUnlabelledDataset, self).__init__()
        self.file_path_getter = file_path_getter

    def __getitem__(self, idx):
        '''Retrieve raw data from disk

        Args:
            idx: index to retrieve

        Returns:
            raw_data (dict)

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]

        for id in idx:
            path = self.file_path_getter(id)
            print (path)
            raise RuntimeError

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
    def __init__(self, path_to_folder, filter_on_suffix):
        p = Path(path_to_folder)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(path_to_folder))

        if filter_on_suffix is None:
            filter_condition = '*'
        else:
            filter_condition = '*.{}'.format(filter_on_suffix)
        file_paths = list(p.glob(filter_condition))

        super(AudioPlainWAVData, self).__init__(lambda idx: file_paths[idx])

class AudioMSSNSDDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self):
        self._instance = NoisySpeechLabelledData(None)

class AudioPlainWAVDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_folder, filter_on_suffix='wav'):
        self._instance = AudioPlainWAVData(path_to_folder=path_to_folder,
                                           filter_on_suffix=filter_on_suffix)

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
