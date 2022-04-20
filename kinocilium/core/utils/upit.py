'''Bla bla

'''
import torch

class UPIT(object):
    '''The Utterance-Level Permutation Invariant Training method to determine loss

    Reference: arXiv: 1703.06284v2

    '''
    def __init__(self, criterion,
                 permutor_model_args=(1,0,2),
                 permutor_ref_args=(1,0,2),
                 select_lower=True,
                 criterion_kwargs={}):
        raise NotImplementedError('The general way to determine what loss to return yet to be implemented')

        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.permutor_model_args = permutor_model_args
        self.permutor_ref_args = permutor_ref_args
        self.select_lower = select_lower

    def __call__(self, model_ts, ref_ts):
        for k_model, model_t in enumerate(model_ts.permute(*self.permutor_model_args)):
            for k_ref, ref_t in enumerate(ref_ts.permute(*self.permutor_ref_args)):
                loss = self.criterion(model_t, ref_t, **self.criterion_kwargs)

