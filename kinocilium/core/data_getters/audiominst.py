from pathlib import Path
from kinocilium.core.data_getters._getters import _AudioLabelledDataset

class AudioMINSTData(_AudioLabelledDataset):
    '''Dataset for labelled audioMINST data

    Args:
        path_to_folder (str): path to folder with audio files
        file_pattern (str): Unix style file selection for which files in folder to include
        read_metadata (bool): if metadata for the files should be read and returned; if not `None` returned
        slice_size (int): if not `None` it describes the number of frames that each slice of data should contain,
            otherwise all frames of each data point are returned

    '''
    def __init__(self, path_to_folder,
                 read_metadata=True,
                 slice_size=None):
        p = Path(path_to_folder)

        super(AudioMINSTData, self).__init__()
