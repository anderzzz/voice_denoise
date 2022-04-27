'''Custom loss criteria

'''
import torch
from torch import nn

def _batchwise_dot(u, v, n_batch):
    return torch.squeeze(torch.bmm(u.view(n_batch, 1, -1), v.view(n_batch, -1, 1)))

def _cmp_s_starget(s_estimate, s_actual, n_batch):
    dot_est_actual = _batchwise_dot(s_estimate, s_actual, n_batch)
    dot_actual_actual = torch.pow(s_actual, 2).sum(dim=1)
    proj_length = dot_est_actual / dot_actual_actual
    return s_actual * proj_length.view(n_batch, -1)

def _cmp_e_noise(s_estimate, s_target, n_batch):
    return s_estimate - s_target

def _cmp_zero_mean(s, n_batch, dim=1):
    return torch.sub(s, torch.mean(s, dim=dim).view(n_batch, -1))

class Source2ArtifactRatio(nn.Module):
    '''Bla bla

    '''
    def __init__(self):
        super(Source2ArtifactRatio, self).__init__()

        raise NotImplementedError

    def forward(self, s_estimate, s_actual):
        return None

class Source2DistortionRatio(nn.Module):
    '''Bla bla

    '''
    def __init__(self):
        super(Source2DistortionRatio, self).__init__()

    def forward(self, s_estimate, s_actual):
        assert s_estimate.shape == s_actual.shape
        n_batch = s_estimate.shape[0]

        s_target = _cmp_s_starget(s_estimate, s_actual, n_batch)
        e_noise = _cmp_e_noise(s_estimate, s_target, n_batch)
        s_power = torch.pow(s_target, 2).sum(dim=1)
        e_power = torch.pow(e_noise, 2).sum(dim=1)

        return 10.0 * torch.log10(s_power) - 10.0 * torch.log10(e_power)

class Source2InterferenceRatio(nn.Module):
    '''Bla bla

    '''
    def __init__(self):
        super(Source2InterferenceRatio, self).__init__()

        raise NotImplementedError

    def forward(self, s_estimate, s_actual):
        return None

class ScaleInvariantSource2DistortionRatio(Source2DistortionRatio):
    '''Bla bla

    '''
    def __init__(self):
        super(ScaleInvariantSource2DistortionRatio, self).__init__()

    def forward(self, s_estimate, s_actual):
        assert s_estimate.shape == s_actual.shape
        n_batch = s_estimate.shape[0]

        s_estimate_si = _cmp_zero_mean(s_estimate, n_batch)
        s_actual_si = _cmp_zero_mean(s_actual, n_batch)

        return super().forward(s_estimate_si, s_actual_si)
