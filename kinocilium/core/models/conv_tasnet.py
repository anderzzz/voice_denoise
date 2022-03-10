'''Model for speech separation. Adopted from "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
Masking for Speech Separation" from 2019 by Luo and Mesgarani

arXiv source: https://arxiv.org/abs/1809.07454

Written by: Anders Ohrn, March 2022

'''
import torch
from torch import nn

class ConvTasNet(nn.Module):
    '''Bla bla

    '''
    def __init__(self):
        super(ConvTasNet, self).__init__()

        pass

    def forward(self, x):
        '''Bla bla

        '''
        raise NotImplementedError
        return None

class ConvTasNetModelBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self):
        self._instance = ConvTasNet()
        return self._instance