'''Bla bla

'''
import sys
import abc
import torch

class ReporterInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Reporter subclasses.
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'append') and
                callable(subclass.append) and
                hasattr(subclass, 'report') and
                callable(subclass.report))

    @abc.abstractmethod
    def reset(self):
        '''Reset reporter before loop over training data set'''
        raise NotImplementedError

    @abc.abstractmethod
    def append(self, descriptor, loss, prediction, data_inputs, epoch, k_batch, minibatch_size):
        '''Append data for future reporting'''
        raise NotImplementedError

    @abc.abstractmethod
    def report(self):
        '''Report progress and performance on training'''
        raise NotImplementedError


class ReporterClassification(ReporterInterface):
    '''Bla bla

    '''
    def __init__(self,
                 dataset_size,
                 report_level='high',
                 f_out=sys.stdout,
                 append_from_inputs=0):
        self.dataset_size = dataset_size
        self._report_level_map = {'low' : 0, 'high' : 1, 'very high' : 2}
        try:
            self.report_level = self._report_level_map[report_level]
        except KeyError:
            raise ValueError('Unknown report level encountered: {}'.format(report_level))
        self.f_out = f_out
        if callable(append_from_inputs):
            self._input_appendor = append_from_inputs
        elif isinstance(append_from_inputs, int):
            self._input_appendor = lambda x: x[append_from_inputs]
        elif append_from_inputs is None:
            self._input_appendor = lambda x: None
        else:
            raise ValueError('Unknown type {} for the `append_from_inputs` parameter'.format(type(append_from_inputs)))

        self.loss_data = []
        self.n_instances = 0

    def reset(self):
        self.loss_data = []
        self.n_instances = 0
        if self.report_level >= self._report_level_map['very high']:
            print('New iteration begins...',
                  file=self.f_out)

    def append(self, descriptor, loss, prediction, data_inputs, epoch, k_batch, minibatch_size):
        self.loss_data.append((descriptor,
                               loss,
                               prediction,
                               self._input_appendor(data_inputs),
                               epoch, k_batch, minibatch_size))
        self.n_instances += minibatch_size
        if self.report_level >= self._report_level_map['high']:
            print ('In epoch {0}, processed data: {1:.1f}%'.format(epoch, 100.0 * self.n_instances / self.dataset_size),
                   file=self.f_out)

    def report(self):
        if self.report_level >= self._report_level_map['low']:
            print ('In epoch {}, reporting:'.format(self.loss_data[0][4]), file=self.f_out)

            tot_data = sum([x[6] for x in self.loss_data])
            loss_agg = sum([x[1] * x[6] for x in self.loss_data]) / tot_data
            print ('    loss: {}'.format(loss_agg))
            if not any([x[3] is None for x in self.loss_data]):
                p_corr = 100.0 * sum([torch.sum(x[2] == x[3]) for x in self.loss_data]) / tot_data
                print ('    correct prediction: {}%'.format(p_corr), file=self.f_out)


class ReporterRunningLoss(ReporterInterface):
    '''Bla bla

    '''
    def __init__(self, dataset_size, report_level='high', f_out=sys.stdout):
        self.dataset_size = dataset_size
        self._report_level_map = {'low' : 0, 'high' : 1, 'very high' : 2}
        try:
            self.report_level = self._report_level_map[report_level]
        except KeyError:
            raise ValueError('Unknown report level encountered: {}'.format(report_level))
        self.f_out = f_out
        self.loss_data = []
        self.n_instances = 0

    def reset(self):
        self.loss_data = []
        self.n_instances = 0
        if self.report_level >= self._report_level_map['very high']:
            print('New iteration begins...',
                  file=self.f_out)

    def append(self, descriptor, loss, prediction, data_inputs, epoch, k_batch, minibatch_size):
        self.loss_data.append((descriptor, loss, epoch, k_batch, minibatch_size))
        self.n_instances += minibatch_size
        if self.report_level >= self._report_level_map['high']:
            print ('In epoch {0}, processed data: {1:.1f}%'.format(epoch, 100.0 * self.n_instances / self.dataset_size),
                   file=self.f_out)

    def report(self):
        total_loss = sum([x[0] * x[2] for x in self.loss_data]) / self.dataset_size
        if self.report_level >= self._report_level_map['low']:
            print ('At end of epoch {0}, running loss: {1:.3f}'.format(self.loss_data[0][1], total_loss),
                   file=self.f_out)


class ReporterNull(ReporterInterface):
    def __init__(self):
        pass
    def reset(self):
        pass
    def append(self, *args, **kwargs):
        pass
    def report(self):
        pass

