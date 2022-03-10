
from pathlib import Path
from kinocilium.core.data_getters._getters import _AudioUnlabelledDataset

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

class AudioPlainWAVDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_folder, file_pattern='*.wav', read_metadata=True):
        self._instance = AudioPlainWAVData(path_to_folder=path_to_folder,
                                           file_pattern=file_pattern,
                                           read_metadata=read_metadata)

        return self._instance