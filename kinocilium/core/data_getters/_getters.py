'''Parent classes for the audio dataset getters

'''
import torchaudio
from torch.utils.data import Dataset

def _torchaudio_meta_2_dict(obj):
    '''Given an AudioMetaData object from torchaudio, it is converted into dictionary

    Purpose of this function is to ensure that DataLoader can collate metadata. It requires some hard-coding of what
    the attribute names are. This can be inspected in the documentation at
    https://pytorch.org/audio/stable/_modules/torchaudio/backend/common.html#AudioMetaData

    '''
    TORCHAUDIO_METADATA_KEYS = {'sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding'}
    ret = {}
    for obj_attr in TORCHAUDIO_METADATA_KEYS:
        ret[obj_attr] = getattr(obj, obj_attr)
    return ret

class _UnChunkedGetter(object):
    '''Audio files data is retrieved from disk in an unchunked manner, that is, the whole file content is retrieved
    regardless of length.

    This should not be used where data is to be batched with the PyTorch DataLoader, since it does not allow for
    ragged batches.

    Args:
        file_path_getter (Callable): function that given index returns file path
        len_file_paths (int): number of file paths that are available
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned

    '''
    def __init__(self, file_path_getter, len_file_paths, read_metadata):
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
        waveform, sample_rate = torchaudio.load(path)
        ret['waveform'] = waveform
        ret['sample_rate'] = sample_rate
        
        if self.read_metadata:
            metadata = _torchaudio_meta_2_dict(torchaudio.info(path))
            ret['metadata'] = metadata

        return ret

class _ChunkedGetter(object):
    '''Audio files data is retrieved from disk in a chunked manner, that is, the file content is retrieved in chunks
    of defined size

    This can be used where data is to be batched with the PyTorch DataLoader. The chunking is done such that any
    remainder is discarded after the integer number of chunks have been created.

    Args:
        file_path_getter (Callable): function that given index returns file path
        len_file_paths (int): number of file paths that are available
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned
        slice_size (int): number of frames in each slice

    '''
    def __init__(self, file_path_getter, len_file_paths, read_metadata, slice_size):
        self.slice_size = slice_size
        self.read_metadata = read_metadata

        self.chunk_map = []
        for fp_idx in range(len_file_paths):
            path = file_path_getter(fp_idx)
            metadata = torchaudio.info(path)
            n_chunks = metadata.num_frames // slice_size
            chunk_map = [(path, k_chunk * slice_size) for k_chunk in range(n_chunks)]
            self.chunk_map += chunk_map

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        '''Retrieve raw data from disk

        Args:
            idx: index to retrieve

        Returns:
            raw_data (dict)

        '''
        ret = {}

        path, frame_offset = self.chunk_map[idx]
        waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=self.slice_size)
        ret['waveform'] = waveform
        ret['sample_rate'] = sample_rate

        if self.read_metadata:
            metadata = _torchaudio_meta_2_dict(torchaudio.info(path))
            metadata['num_frames'] = self.slice_size
            ret['metadata'] = metadata

        return ret

class _AudioUnlabelledDataset(Dataset):
    '''Parent class for unlabelled audio data of the PyTorch Dataset format

    Args:
        file_path_getter (Callable): function that given index returns file path
        len_file_paths (int): number of file paths that are available
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned

    '''
    def __init__(self, file_path_getter, len_file_paths, read_metadata, slice_size):
        super(_AudioUnlabelledDataset, self).__init__()

        self.read_metadata = read_metadata
        if slice_size is None:
            self._getter = _UnChunkedGetter(file_path_getter, len_file_paths, read_metadata)

        elif isinstance(slice_size, int):
            self._getter = _ChunkedGetter(file_path_getter, len_file_paths, read_metadata, slice_size)

        else:
            raise ValueError('Non-allowed type for `slice_size`. Must be `int` or `None`')

    def __len__(self):
        return self._getter.__len__()

    def __getitem__(self, idx):
        return self._getter.__getitem__(idx)

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
            metadata = _torchaudio_meta_2_dict(torchaudio.info(path_file))
            metadata_counterpart = _torchaudio_meta_2_dict(torchaudio.info(path_file_counterpart))

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