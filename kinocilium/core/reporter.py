'''Bla bla

'''
import sys
import abc

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
    def append(self, descriptor, loss, prediction, epoch, k_batch, minibatch_size):
        '''Append data for future reporting'''
        raise NotImplementedError

    @abc.abstractmethod
    def report(self):
        '''Report progress and performance on training'''
        raise NotImplementedError


class ReporterClassification(ReporterInterface):
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

    def append(self, descriptor, loss, prediction, epoch, k_batch, minibatch_size):
        self.loss_data.append((loss, epoch, k_batch, minibatch_size))
        self.n_instances += minibatch_size
        if self.report_level >= self._report_level_map['high']:
            print ('In epoch {0}, processed data: {1:.1f}%'.format(epoch, 100.0 * self.n_instances / self.dataset_size),
                   file=self.f_out)

    def report(self):
        raise NotImplementedError
        pass


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

    def append(self, descriptor, loss, prediction, epoch, k_batch, minibatch_size):
        self.loss_data.append((loss, epoch, minibatch_size))
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

