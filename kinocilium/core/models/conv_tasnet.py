'''Model for speech separation. Adopted from "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
Masking for Speech Separation" from 2019 by Luo and Mesgarani

arXiv source: https://arxiv.org/abs/1809.07454

Written by: Anders Ohrn, March 2022

'''
import torch
from torch import nn

from kinocilium.core.models._blocks import PointwiseConv1d, DepthWiseConv1d

NORM_EPS = 1e-08

class SeparableConv1DNormBlockSkipRes(nn.Module):
    '''Bla bla

    '''
    INNER_1D_CONV_BLOCK_ORDER = ['pointwise convolution',
                                 'first non-linearity',
                                 'first normalization',
                                 'depthwise convolution',
                                 'second non-linearity',
                                 'second normalization']
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels_skip,
                 out_channels_residual,
                 kernel_size,
                 dilation,
                 padding,
                 bias,
                 device=None,
                 dtype=None):
        super(SeparableConv1DNormBlockSkipRes, self).__init__()

        self.separable_conv1d_inner = nn.ModuleDict()
        self.separable_conv1d_inner['first non-linearity'] = nn.PReLU(device=device, dtype=dtype)
        self.separable_conv1d_inner['second non-linearity'] = nn.PReLU(device=device, dtype=dtype)
        self.separable_conv1d_inner['pointwise convolution'] = \
            PointwiseConv1d(in_channels=in_channels,
                            out_channels=hidden_channels,
                            bias=bias,
                            device=device,
                            dtype=dtype)
        self.separable_conv1d_inner['depthwise convolution'] = \
            DepthWiseConv1d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            bias=bias,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            stride=1,
                            padding=padding,
                            device=device,
                            dtype=dtype)
        self.separable_conv1d_inner['first normalization'] = nn.GroupNorm(num_groups=1,
                                                                          num_channels=hidden_channels,
                                                                          eps=NORM_EPS,
                                                                          device=device,
                                                                          dtype=dtype)
        self.separable_conv1d_inner['second normalization'] = nn.GroupNorm(num_groups=1,
                                                                           num_channels=hidden_channels,
                                                                           eps=NORM_EPS,
                                                                           device=device,
                                                                           dtype=dtype)
        assert set(self.separable_conv1d_inner.keys()) == set(self.INNER_1D_CONV_BLOCK_ORDER)

        self.skip_connection = PointwiseConv1d(in_channels=hidden_channels,
                                               out_channels=out_channels_skip,
                                               device=device,
                                               dtype=dtype)
        self.residual = PointwiseConv1d(in_channels=hidden_channels,
                                        out_channels=out_channels_residual,
                                        device=device,
                                        dtype=dtype)

    def forward(self, x):
        for key in self.INNER_1D_CONV_BLOCK_ORDER:
            x = self.separable_conv1d_inner[key](x)
        x_skip = self.skip_connection(x)
        x_res = self.residual(x)

        return x_skip, x_res


class SeparationBlock(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 n_repeats,
                 n_blocks,
                 in_channels,
                 n_bottleneck_channels,
                 n_skip_channels,
                 n_hidden_channels,
                 n_separation_conv_kernel_width,
                 conv_dilator,
                 conv_bias,
                 n_sources,
                 device=None,
                 dtype=None):
        super(SeparationBlock, self).__init__()

        self.n_repeats = n_repeats
        self.n_blocks = n_blocks

        #
        # Initial step in separation module where encoded tensor linearly transformed into hidden channels
        # in a pointwise fashion
        self.layer_init = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=NORM_EPS),
            PointwiseConv1d(in_channels=in_channels,
                            out_channels=n_bottleneck_channels,
                            device=device,
                            dtype=dtype)
        )

        #
        # The separable convolution blocks in a number of repeats
        if conv_dilator == 'exponential':
            self._dilator = lambda x: 2**x
        elif conv_dilator is None:
            self._dilator = lambda x: 1
        elif callable(conv_dilator):
            self._dilator = conv_dilator
        else:
            raise ValueError('The convolution dilator "{}" not recognized'.format(self.conv_dilator))

        self.layer_modules = nn.ModuleDict()
        for k_repeat in range(n_repeats):
            for k_block in range(n_blocks):
                self.layer_modules[self._make_key(k_repeat, k_block)] = \
                    SeparableConv1DNormBlockSkipRes(in_channels=n_bottleneck_channels,
                                                    hidden_channels=n_hidden_channels,
                                                    out_channels_skip=n_skip_channels,
                                                    out_channels_residual=n_bottleneck_channels,
                                                    kernel_size=n_separation_conv_kernel_width,
                                                    dilation=self._dilator(k_block),
                                                    padding=self._dilator(k_block),
                                                    bias=conv_bias,
                                                    device=device,
                                                    dtype=dtype)

        #
        # Concluding activation, transformation in a pointwise fashion and mapping into range 0.0-1.0
        self.layer_post = nn.Sequential(
            nn.PReLU(),
            PointwiseConv1d(in_channels=n_bottleneck_channels,
                           out_channels=n_sources * in_channels,
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
        n_encoder_filters (int): Number of encoder filters. N in paper.
        n_encoder_kernel_width (int): Length of overlapping audio segments. L in paper.
        n_repeats (int): Number of convolution layer repeats of separation module. R in paper.
        n_blocks (int): Number of convolution blocks in each layer of separation module. X in paper.
        n_bottleneck_channels (int): Number of channels in residual path of separation module. B in paper.
        n_skip_channels (int): Number of channels in skip path of separation module. Sc in paper.
        n_hidden_channels (int): Number of hidden channels in the separation module. H in paper.
        device (str):
        dtype (str):

    '''
    def __init__(self,
                 in_channels,
                 n_sources,
                 n_encoder_filters,
                 n_encoder_kernel_width,
                 p_encoder_window_overlap,
                 n_repeats,
                 n_blocks,
                 n_bottleneck_channels,
                 n_skip_channels,
                 n_hidden_channels,
                 n_separation_conv_kernel_width,
                 conv_dilator,
                 conv_bias,
                 device=None,
                 dtype=None):
        super(ConvTasNet, self).__init__()

        if in_channels != 1:
            raise NotImplementedError('ConvTasNet only implemented for mono audio, in_channels must be 1')
        else:
            self.in_channels = in_channels

        self.n_encoder_filters = n_encoder_filters
        self.n_encoder_window = n_encoder_kernel_width
        self.p_encoder_window_overlap = p_encoder_window_overlap
        stride = int(self.n_encoder_window * p_encoder_window_overlap / 100)
        if self.n_encoder_window % 2 == 1:
            padding = (self.n_encoder_window - 1) // 2
        else:
            raise ValueError('The encoder window length must be odd, so not {}'.format(self.n_encoder_window))

        self.encoder = nn.Conv1d(in_channels=self.in_channels,
                                 out_channels=self.n_encoder_filters,
                                 kernel_size=self.n_encoder_window,
                                 bias=False,
                                 stride=stride,
                                 padding=padding,
                                 padding_mode='zeros',
                                 device=device,
                                 dtype=dtype)
        self.decoder = nn.ConvTranspose1d(in_channels=self.n_encoder_filters,
                                          out_channels=self.in_channels,
                                          kernel_size=self.n_encoder_window,
                                          bias=False,
                                          stride=stride,
                                          device=device,
                                          dtype=dtype)

        if not n_sources > 1:
            raise ValueError('Number of separable sources has to exceed 1')
        else:
            self.n_sources = n_sources
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.n_bottleneck_filters = n_bottleneck_channels
        self.n_skip_channels = n_skip_channels
        self.n_hidden_channels = n_hidden_channels
        self.n_separation_conv_kernel_width = n_separation_conv_kernel_width
        self.conv_dilator = conv_dilator
        self.conv_bias = conv_bias
        self.separation_block = SeparationBlock(n_repeats=self.n_repeats,
                                                n_blocks=self.n_blocks,
                                                in_channels=self.n_encoder_filters,
                                                n_bottleneck_channels=self.n_bottleneck_filters,
                                                n_skip_channels=self.n_skip_channels,
                                                n_hidden_channels=self.n_hidden_channels,
                                                n_separation_conv_kernel_width=self.n_separation_conv_kernel_width,
                                                conv_dilator=self.conv_dilator,
                                                conv_bias=self.conv_bias,
                                                n_sources=self.n_sources,
                                                device=device,
                                                dtype=dtype)

    def forward(self, x):
        '''Bla bla

        '''
        batch_size = x.size(0)

        x_enc = self.encoder(x)
        mask = self.separation_block(x_enc)
        mask = mask.view(batch_size, self.n_sources, self.n_encoder_filters, -1)
        masked_output = x_enc.unsqueeze(1) * mask
        masked_output = masked_output.view(batch_size * self.n_sources, self.n_encoder_filters, -1)
        x_decode_masked = self.decoder(masked_output)
        x_decode_masked = x_decode_masked.view(batch_size, self.n_sources, -1)
        # TODO: tweak the final slicing of x_decode_masked so it has the same number of data as input. Something with stride and padding

        return x_decode_masked

class ConvTasNetModelBuilder(object):
    def __init__(self):
        self._instance = None
    def __call__(self, in_channels=1, n_sources=2,
                 n_encoder_filters=512, n_encoder_kernel_width=39, p_encoder_window_overlap=50.0,
                 n_repeats=2, n_blocks=7,
                 n_bottleneck_channels=128, n_skip_channels=128, n_hidden_channels=256,
                 n_separation_conv_kernel_width=3,
                 conv_dilator='exponential', conv_bias=True,
                 device=None, dtype=None):
        self._instance = ConvTasNet(in_channels=in_channels,
                                    n_sources=n_sources,
                                    n_encoder_filters=n_encoder_filters,
                                    n_encoder_kernel_width=n_encoder_kernel_width,
                                    p_encoder_window_overlap=p_encoder_window_overlap,
                                    n_repeats=n_repeats,
                                    n_blocks=n_blocks,
                                    n_bottleneck_channels=n_bottleneck_channels,
                                    n_skip_channels=n_skip_channels,
                                    n_hidden_channels=n_hidden_channels,
                                    n_separation_conv_kernel_width=n_separation_conv_kernel_width,
                                    conv_dilator=conv_dilator,
                                    conv_bias=conv_bias,
                                    device=device,
                                    dtype=dtype)

        if not 'init_kwargs' in self._instance.__dir__():
            self._instance.init_kwargs = {
                                          'in_channels' : in_channels,
                                          'n_sources' : n_sources,
                                          'n_encoder_filters' : n_encoder_filters,
                                          'n_encoder_kernel_width' : n_encoder_kernel_width,
                                          'p_encoder_window_overlap' : p_encoder_window_overlap,
                                          'n_repeats' : n_repeats,
                                          'n_blocks' : n_blocks,
                                          'n_bottleneck_channels' : n_bottleneck_channels,
                                          'n_skip_channels' : n_skip_channels,
                                          'n_hidden_channels' : n_hidden_channels,
                                          'n_separation_conv_kernel_width' : n_separation_conv_kernel_width,
                                          'conv_dilator' : conv_dilator,
                                          'conv_bias' : conv_bias,
                                          'device' : device,
                                          'dtype' : dtype
                                          }

        return self._instance