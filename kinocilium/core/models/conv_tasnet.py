'''Model for speech separation. Adopted from "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
Masking for Speech Separation" from 2019 by Luo and Mesgarani

arXiv source: https://arxiv.org/abs/1809.07454

Written by: Anders Ohrn, March 2022

'''
import torch
from torch import nn

from _blocks import PointwiseConv1D, DepthWiseConv1D

class _SeparableConv1DBasicBlock(nn.Module):
    '''Separable 1D convolution block with normalization and activation

    This is the
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 stride=1,
                 padding=0,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super(_SeparableConv1DBasicBlock, self).__init__()

        self.separable_conv1d_basic_block = nn.Sequential(
            PointwiseConv1D(in_channels=in_channels,
                            out_channels=out_channels,
                            bias=bias,
                            device=device,
                            dtype=dtype),
            nn.PReLU(device=device,
                     dtype=dtype),
            NORM,
            DepthWiseConv1D(in_channels=out_channels,
                            out_channels=out_channels,
                            bias=bias,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            stride=stride,
                            padding=padding,
                            padding_mode=padding_mode,
                            device=device,
                            dtype=dtype),
        nn.PReLU(device=device,
                 dtype=dtype),
        NORM
        )

    def forward(self, x):
        return self.separable_conv1d_basic_block(x)

class SeparableConv1DNormBlockSkipRes(nn.Module):
    '''Bla bla

    '''
    def __init__(self):
        super(SeparableConv1DNormBlockSkipRes, self).__init__()

        self.separable_conv1d_inner = _SeparableConv1DBasicBlock()
        self.skip_connection = PointwiseConv1D()
        self.residual = PointwiseConv1D()

    def forward(self, x):
        x = self.separable_conv1d_inner(x)
        x_skip = self.skip_connection(x)
        x_res = self.residual(x)

        return x_skip, x_res


class SeparationBlock(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 n_repeats,
                 n_blocks,
                 device=None,
                 dtype=None):
        super(SeparationBlock, self).__init__()

        self.n_repeats = n_repeats
        self.n_blocks = n_blocks

        self.layer_init = nn.Sequential(
            nn.LayerNorm(),
            PointwiseConv1D()
        )

        self.layer_modules = nn.ModuleDict({})
        for k_repeat in range(self.n_repeats):
            for k_block in range(self.n_blocks):
                one_d = nn.Conv1d(in_channels=XX,
                                  out_channels=XX,
                                  kernel_size=XX,
                                  dilation=dd)

                self.layer_modules[self._make_key(k_repeat, k_block)] = one_d

        self.layer_post = nn.Sequential(
            nn.PReLU(),
            PointwiseConv1D(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_res_accumulated = None

        x_init = self.layer_init(x)
        x_skip = x_init
        for k_repeat in range(self.n_repeats):
            for k_block in range(self.n_blocks):
                x_skip, x_res = self.layer_modules[self._make_key(k_repeat, k_block)](x_skip)

                if x_res_accumulated is None:
                    x_res_accumulated = x_res
                else:
                    x_res_accumulated += x_res

        x_post = self.layer_post(x_res_accumulated)

        return x_post

    def _make_key(self, k_repeat, k_block):
        return 'middle_conv1d_repeat_{}_block_{}'.format(k_repeat, k_block)

class ConvTasNet(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 in_channels,
                 n_encoder_filters,
                 n_sources,
                 filter_length,
                 n_repeats,
                 n_blocks,
                 device=None,
                 dtype=None):
        super(ConvTasNet, self).__init__()

        self.encoder = nn.Conv1d(in_channels=in_channels,
                                 out_channels=n_encoder_filters,
                                 kernel_size=filter_length,
                                 device=device,
                                 dtype=dtype)
        self.separation_block = SeparationBlock(n_repeats=n_repeats,
                                                n_blocks=n_blocks,
                                                device=device,
                                                dtype=dtype)
        self.decoder = nn.ConvTranspose1d(in_channels=xxx,
                                          out_channels=n_sources,
                                          kernel_size=yyy,
                                          device=device,
                                          dtype=dtype)

    def forward(self, x):
        '''Bla bla

        '''
        x_enc = self.encoder(x)
        x_sep = self.separation_block(x_enc)
        x_final = self.decoder(x_sep)

        return x_final

class ConvTasNetModelBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self):
        self._instance = ConvTasNet()
        return self._instance