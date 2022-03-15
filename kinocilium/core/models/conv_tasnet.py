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
                 n_bottleneck_channels,
                 n_hidden_channels,
                 n_sources,
                 device=None,
                 dtype=None):
        super(SeparationBlock, self).__init__()

        self.n_repeats = n_repeats
        self.n_blocks = n_blocks
        self.n_bottleneck_channels = n_bottleneck_channels
        self.n_hidden_channels = n_hidden_channels
        self.n_sources = n_sources

        #
        # Initial step in separation module where encoded tensor linearly transformed into hidden channels
        # in a pointwise fashion
        self.layer_init = nn.Sequential(
            nn.LayerNorm(),
            nn.Conv1d(in_channels=self.n_bottleneck_channels,
                      out_channels=self.n_hidden_channels,
                      kernel_size=1,
                      device=device,
                      dtype=dtype)
        )

        #
        # The separable convolution blocks in a number of repeats
        self.layer_modules = nn.ModuleDict({})
        for k_repeat in range(self.n_repeats):
            for k_block in range(self.n_blocks):
                self.layer_modules[self._make_key(k_repeat, k_block)] = \
                    SeparableConv1DNormBlockSkipRes()

        #
        # Concluding activation, transformation in a pointwise fashion and mapping into range 0.0-1.0
        self.layer_post = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(in_channels=self.n_bottleneck_channels,
                      out_channels=self.n_sources * self.n_bottleneck_channels,
                      kernel_size=1,
                      device=device,
                      dtype=dtype),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_accumulated = None

        x_init = self.layer_init(x)
        x_out = x_init
        for k_repeat in range(self.n_repeats):
            for k_block in range(self.n_blocks):
                x_skip, x_res = self.layer_modules[self._make_key(k_repeat, k_block)](x_out)

                x_out += x_res
                if x_accumulated is None:
                    x_accumulated = x_skip
                else:
                    x_accumulated += x_skip

        x_post = self.layer_post(x_accumulated)

        return x_post

    def _make_key(self, k_repeat, k_block):
        return 'middle_conv1d_repeat_{}_block_{}'.format(k_repeat, k_block)

class ConvTasNet(nn.Module):
    '''Bla bla

    Args:
        in_channels (int): Number of channels of input audio. Mono audio implies `in_channels=1`.
        n_sources (int): Number of sources to separate input audio into.
        segment_length (int): Length of overlapping audio segments. L in paper.
        n_encoder_filters (int): Number of encoder filters. N in paper.
        n_residual_channels (int): Number of channels in residual path of separation module. B in paper.
        n_skip_channels (int): Number of channels in skip path of separation module. Sc in paper.
        n_repeats (int): Number of convolution layer repeats of separation module. R in paper.
        n_blocks (int): Number of convolution blocks in each layer of separation module. X in paper.


    '''
    def __init__(self,
                 in_channels,
                 n_sources,
                 n_encoder_filters,
                 n_encoder_kernel_width,
                 n_residual_channels,
                 n_skip_channels,
                 filter_length,
                 n_repeats,
                 n_blocks,
                 stride,
                 device=None,
                 dtype=None):
        super(ConvTasNet, self).__init__()

        if in_channels != 1:
            raise NotImplementedError('ConvTasNet only implemented for mono audio, in_channels must be 1')
        else:
            self.in_channels = in_channels

        if not n_sources > 1:
            raise ValueError('Number of separable sources has to exceed 1')
        else:
            self.n_sources = n_sources

        self.n_encoder_filters = n_encoder_filters
        self.n_encoder_window = n_encoder_kernel_width
        if self.n_encoder_window % 2 == 1:
            padding = ((self.n_encoder_window - 1) // 2, (self.n_encoder_window - 1) // 2)
        else:
            padding = (self.n_encoder_window // 2, self.n_encoder_window // 2 - 1)

        self.encoder = nn.Conv1d(in_channels=self.in_channels,
                                 out_channels=self.n_encoder_filters,
                                 kernel_size=self.n_encoder_window,
                                 bias=False,
                                 stride=1,
                                 padding=padding,
                                 padding_mode='zeros',
                                 device=device,
                                 dtype=dtype)
        self.decoder = nn.ConvTranspose1d(in_channels=self.n_encoder_filters,
                                          out_channels=self.in_channels,
                                          kernel_size=self.n_encoder_window,
                                          bias=False,
                                          stride=1,
#                                          padding=padding,
#                                          padding_mode='zeros',
                                          device=device,
                                          dtype=dtype)

        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.separation_block = SeparationBlock(n_repeats=self.n_repeats,
                                                n_blocks=self.n_blocks,
                                                n_bottleneck_channels=self.n_encoder_filters,
                                                n_hidden_channels=XXX,
                                                n_sources=self.n_sources,
                                                device=device,
                                                dtype=dtype)

    def forward(self, x):
        '''Bla bla

        '''
        x_enc = self.encoder(x)
        mask = self.separation_blocks(x_enc)
        raise NotImplementedError
#            x_source = self.decoder(torch.mul(mask, x_enc))

        return separated_sources

class ConvTasNetModelBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self):
        self._instance = ConvTasNet()
        return self._instance