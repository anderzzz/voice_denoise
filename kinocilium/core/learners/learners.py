'''Learner modules for various types of audio tasks

Written by: Anders Ohrn, March 2022

'''
import sys

from kinocilium.core.learners._learner import _Learner


class LearnerPairedAudio(_Learner):
    '''Bla bla

    '''
    def __init__(self, run_label='Learner Run Label',
                       random_seed=None,
                       f_out=sys.stdout,
                       save_tmp_name='model_in_training',
                       num_workers=0,
                       deterministic=True):

        super(LearnerPairedAudio, self).__init__(run_label=run_label,
                                                 random_seed=random_seed,
                                                 f_out=f_out,
                                                 save_tmp_name=save_tmp_name,
                                                 num_workers=num_workers,
                                                 deterministic=deterministic)

        def load_model(self, path):
            pass

        def save_model(self, path):
            pass

        def train(self, n_epochs):
            '''Train model

            Args:
                n_epochs (int): Number of training epochs

            '''
            pass

        def validate(self):
            pass

class LearnerPairedAudioBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self):
        self._instance = LearnerPairedAudio()
        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {}

        return self._instance