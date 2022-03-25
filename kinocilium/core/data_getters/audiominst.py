from pathlib import Path
from kinocilium.core.data_getters._getters import _AudioLabelledDataset

class AudioMINSTData(_AudioLabelledDataset):
    '''Dataset for labelled audioMINST data

    Args:
        path_to_folder (str): path to folder with audio files
        label (str): label type to return along with audio file, either of `digit`, `person` or `repeat`
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned
        slice_size (int): if not `None` it describes the number of frames that each slice of data should contain,
            otherwise all frames of each data point are returned

    '''
    def __init__(self, path_to_folder,
                 label,
                 read_metadata=True,
                 slice_size=None):
        label2index = {'digit' : 0, 'person' : 1, 'repeat' : 2}
        if not label in label2index.keys():
            raise ValueError('The label for AudioMINST must be one of these: {}'.format(list(label2index.keys())))

        p = Path(path_to_folder)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(path_to_folder))

        file_paths = list(p.glob('*'))
        n_file_paths = len(file_paths)

        labels = []
        for file in file_paths:
            file_name_stem_parts = Path(file).stem.split('_')
            labels.append(int(file_name_stem_parts[label2index[label]]))

        super(AudioMINSTData, self).__init__(file_path_getter=lambda idx: file_paths[idx],
                                             label_getter=lambda idx: labels[idx],
                                             len_file_paths=n_file_paths,
                                             read_metadata=read_metadata,
                                             slice_size=slice_size)

class AudioMINSTDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, path_to_folder,
                 label,
                 read_metadata=True,
                 slice_size=None):
        self._instance = AudioMINSTData(path_to_folder=path_to_folder,
                                        label=label,
                                        read_metadata=read_metadata,
                                        slice_size=slice_size)
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {'path_to_folder' : path_to_folder,
                                          'label' : label,
                                          'read_metadata' : read_metadata,
                                          'slice_size' : slice_size}

        return self._instance