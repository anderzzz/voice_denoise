'''Bla bla

'''
import torch

CHECK_SDR = torch.tensor([15.5091, 14.3136, -7.7003], dtype=float)
CHECK_SI_SDR = torch.tensor([6.0357, 13.2480, 4.1114], dtype=float)

from kinocilium.core.loss_criteria import Source2DistortionRatio, ScaleInvariantSource2DistortionRatio

t_estimate = torch.tensor([[1,3,3,4], [5,1,0,1], [7,6,6,0]], dtype=float)
t_actual   = torch.tensor([[2,3,3,5], [5,1,1,1], [3,2,0,7]], dtype=float)

sdr_criterion = Source2DistortionRatio()
si_sdr_criterion = ScaleInvariantSource2DistortionRatio()

loss1 = sdr_criterion(t_estimate, t_actual)
loss2 = si_sdr_criterion(t_estimate, t_actual)

assert torch.abs(torch.sum(loss1 - CHECK_SDR)) / 4 < 1e-4
assert torch.abs(torch.sum(loss2 - CHECK_SI_SDR)) / 4 < 1e-4
