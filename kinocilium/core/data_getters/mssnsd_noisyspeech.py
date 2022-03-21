'''Bla bla

'''
from pathlib import Path
import re

from kinocilium.core.data_getters._getters import _AudioPairedDataset

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
                 read_metadata=True,
                 slice_size=None):
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
                                                    read_metadata=read_metadata,
                                                    slice_size=slice_size)

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
                 read_metadata=True,
                 slice_size=None):
        self._instance = MSSNSDNoisySpeechData(path_to_noisyspeech=path_to_noisyspeech,
                                               path_to_cleanspeech=path_to_cleanspeech,
                                               path_to_noise=path_to_noise,
                                               return_clean_counterpart=return_clean_counterpart,
                                               filter_on_snr=filter_on_snr,
                                               read_metadata=read_metadata,
                                               slice_size=slice_size)
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {'path_to_noisyspeech' : path_to_noisyspeech,
                                          'path_to_cleanspeech' : path_to_cleanspeech,
                                          'path_to_noise' : path_to_noise,
                                          'return_clean_counterpart' : return_clean_counterpart,
                                          'filter_on_snr' : filter_on_snr,
                                          'read_metadata' : read_metadata,
                                          'slice_size' : slice_size
                                          }

        return self._instance