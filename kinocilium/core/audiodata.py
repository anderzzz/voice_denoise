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

    Args:
        file_path_getter (Callable): function that given index returns file path
        len_file_paths (int): number of file paths that are available
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned

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

    Args:
        file_paths_getter (Callable): function that given index return to pair of file paths
        len_files_path (int): number of pairs of file paths
        keys_filetype (tuple): tuple of strings with labels for the two types of file paths
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned

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
        else:
            metadata = None
            metadata_counterpart = None

        waveform, sample_rate = torchaudio.load(path_file)
        waveform_counterpart, sample_rate_counterpart = torchaudio.load(path_file_counterpart)

        return {'waveform_{}'.format(self.keys_filetype[0]) : waveform,
                'sample_rate_{}'.format(self.keys_filetype[0]) : sample_rate,
                'metadata_{}'.format(self.keys_filetype[0]) : metadata,
                'waveform_{}'.format(self.keys_filetype[1]) : waveform_counterpart,
                'sample_rate_{}'.format(self.keys_filetype[1]) : sample_rate_counterpart,
                'metadata_{}'.format(self.keys_filetype[1]) : metadata_counterpart}

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
    '''Dataset for unlabelled audio data

    Args:
        path_to_folder (str): path to folder with audio files
        file_pattern (str): Unix style file selection for which files in folder to include
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned

    '''
    def __init__(self, path_to_folder, file_pattern='*.wav',
                 read_metadata=True):
        p = Path(path_to_folder)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(path_to_folder))

        file_paths = list(p.glob(file_pattern))
        n_file_paths = len(file_paths)

        super(AudioPlainWAVData, self).__init__(file_path_getter=lambda idx: file_paths[idx],
                                                len_file_paths=n_file_paths,
                                                read_metadata=read_metadata)

class MSSNSDNoisySpeechData(_AudioPairedDataset):
    '''Dataset for MS-SNSD noisy speech data generated with standard scripts

    The audio files are assumed to have been generated with `noisespeech_synthesizer.py` at the repo:
    https://github.com/microsoft/MS-SNSD. This generates three subfolders: one with the synthesized noisy speech,
    one with the clean speech and one with the noise. The file name for noisy speech is constructed like:
    `noisy14_SNRdb_40.0_clnsp14.wav`. The name defines the noise file that is used (`14` in example), at what
    speech to noise ratio it was synthesizes (`40.0` in example), and what clean speech was used (`14` in example).
    Given this manner of file name construction, the class derives the file paths to the component audio
    files.

    Args:
        path_to_noisyspeech (str):
        path_to_cleanspeech (str):
        path_to_noise (str):
        return_clean_counterpart (bool):
        filter_on_snr (str):
        read_metadata (bool):

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
        '''Method to return the file paths for the paired MS-SNSD audio data

        The method assumes a structure to the filename of the files generated by the MS-SNSD scripts. Given
        a noisy speech file name, the two possible counterparts (clean speech and noise) file names and
        file paths are generated.

        '''
        p_file_noisyspeech = Path(self._file_paths[idx])
        all_numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", p_file_noisyspeech.name)
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

        if self.return_clean_counterpart:
            ret_counterpart = self.p_cleanspeech.joinpath(clean_file_name)
        else:
            ret_counterpart = self.p_noisyspeech.joinpath(noisy_file_name)

        return (p_file_noisyspeech, ret_counterpart)


class AudioMSSNSDDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_noisyspeech, path_to_cleanspeech, path_to_noise,
                 return_clean_counterpart=True, filter_on_snr='[0-9]+\.[0-9]+',
                 read_metadata=True):
        self._instance = MSSNSDNoisySpeechData(path_to_noisyspeech=path_to_noisyspeech,
                                               path_to_cleanspeech=path_to_cleanspeech,
                                               path_to_noise=path_to_noise,
                                               return_clean_counterpart=return_clean_counterpart,
                                               filter_on_snr=filter_on_snr,
                                               read_metadata=True)

        return self._instance

class AudioPlainWAVDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_folder, file_pattern='*.wav', read_metadata=True):
        self._instance = AudioPlainWAVData(path_to_folder=path_to_folder,
                                           file_pattern=file_pattern,
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
            builder: An Audio Data Builder instance

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

