'''Bla bla

'''
# Constant: name of test data subfolder to use for test
DATA_SUBFOLDER = 'data3'

import os
abs_data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), DATA_SUBFOLDER)

from torch.utils.data import DataLoader
import torch
torch.manual_seed(42)

from kinocilium.core.data_getters import factory as factory_data
from kinocilium.core.calibrators import factory as factory_calibrator
from kinocilium.core.calibrators.reporter import ReporterClassification

class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=39, stride=19, padding=19),
            torch.nn.Conv1d(in_channels=3, out_channels=9, kernel_size=39, stride=19, padding=19),
            torch.nn.Conv1d(in_channels=9, out_channels=27, kernel_size=64)
        )
        self.nonlins = torch.nn.Sequential(
            torch.nn.Linear(in_features=27, out_features=27),
            torch.nn.PReLU(),
            torch.nn.Linear(in_features=27, out_features=10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.squeeze(x, dim=-1)
        x = self.nonlins(x)
        return x

def test_simple_init_and_call():
    data = factory_data.create('audio-minst', path_to_folder=abs_data_path, label='digit', read_metadata=False, slice_size=23000)
    model = DummyModel()
    xx= model(torch.unsqueeze(data[0][1]['waveform'], dim=0))

    assert tuple(xx.shape) == (1, 10)

    calibrator = factory_calibrator.create('labelled audio classification',
                                           optimizer_parameters=model.parameters(),
                                           )
    calibrator.train(model, 1, DataLoader(data, batch_size=10))

def test_simple_init_and_call_report():
    data = factory_data.create('audio-minst', path_to_folder=abs_data_path, label='digit', read_metadata=False, slice_size=23000)
    model = DummyModel()
    xx= model(torch.unsqueeze(data[0][1]['waveform'], dim=0))

    assert tuple(xx.shape) == (1, 10)

    calibrator = factory_calibrator.create('labelled audio classification',
                                           optimizer_parameters=model.parameters()
                                           )
    calibrator.train(model=model,
                     n_epochs=1,
                     dataloader=DataLoader(data, batch_size=6),
                     reporter=ReporterClassification(append_from_inputs=0)
                     )



if __name__ == '__main__':
    test_simple_init_and_call()
    test_simple_init_and_call_report()