'''Learner modules for various types of audio tasks

Written by: Anders Ohrn, March 2022

'''
import sys
import torch

from _learner import _Learner, progress_bar

class LearnerPairedAudio(_Learner):
    '''Bla bla

    '''
    def __init__(self, run_label='Learner Run Label',
                       random_seed=None,
                       f_out=sys.stdout,
                       save_tmp_name='model_in_training',
                       dataset_type='plain wav',
                       dataset_kwargs={},
                       loader_batch_size=16,
                       num_workers=0,
                       deterministic=True):

        super(LearnerPairedAudio, self).__init__(run_label=run_label,
                                                 random_seed=random_seed,
                                                 f_out=f_out,
                                                 save_tmp_name=save_tmp_name,
                                                 dataset_type=dataset_type,
                                                 dataset_kwargs=dataset_kwargs,
                                                 loader_batch_size=loader_batch_size,
                                                 num_workers=num_workers,
                                                 deterministic=deterministic)
        self.model = None

        def inject_model(self, nn_module):
            self.model = nn_module

        def load_model(self, path):
            pass

        def save_model(self, path):
            pass

        def train(self, n_epochs):
            '''Train model

            Args:
                n_epochs (int): Number of training epochs

            '''
