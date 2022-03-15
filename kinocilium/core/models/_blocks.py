'''Bla bla

'''
import torch
from torch import nn

class DepthWiseConv1d(nn.Module):
    '''Depthwise 1D convolution. Wrapper over `Conv1d`.

    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super(DepthWiseConv1d, self).__init__()

        self.depthwise_conv = nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode,
                                        device=device,
                                        dtype=dtype)

    def forward(self, x):
        x = self.depthwise_conv(x)

        return x

class PointwiseConv1d(nn.Module):
    '''Pointwise 1D convolution.

    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None):
        super(PointwiseConv1d, self).__init__()

        self.pointwise_conv = nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)

    def forward(self, x):
        x = self.pointwise_conv(x)

        return x

