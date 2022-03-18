'''Parent classes for the audio dataset getters

'''
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
        ret = {}

        path = self.file_path_getter(idx)
        if self.read_metadata:
            metadata = torchaudio.info(path)
            ret['metadata'] = metadata

        waveform, sample_rate = torchaudio.load(path)
        ret['waveform'] = waveform
        ret['sample_rate'] = sample_rate

        return ret


class _AudioLabelledDataset(Dataset):
    '''Parent class for labelled audio data of the PyTorch Dataset format

    '''
    def __init__(self):
        raise NotImplementedError
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

        waveform, sample_rate = torchaudio.load(path_file)
        waveform_counterpart, sample_rate_counterpart = torchaudio.load(path_file_counterpart)

        ret = {'waveform_{}'.format(self.keys_filetype[0]) : waveform,
               'sample_rate_{}'.format(self.keys_filetype[0]) : sample_rate,
               'waveform_{}'.format(self.keys_filetype[1]) : waveform_counterpart,
               'sample_rate_{}'.format(self.keys_filetype[1]) : sample_rate_counterpart}
        if self.read_metadata:
            ret['metadata_{}'.format(self.keys_filetype[0])] = metadata
            ret['metadata_{}'.format(self.keys_filetype[1])] = metadata_counterpart

        return ret