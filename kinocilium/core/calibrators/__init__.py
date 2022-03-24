'''Initialization upon import of `calibrators`

The import generates:
 * Access to `factory`. This is the learner object creation factory. It provides a uniform interface to all
calibrators.

Written by: Anders Ohrn, March 2020

'''
from kinocilium.core._factory import _Factory
from kinocilium.core.calibrators.calibrators import CalibratorPairedAudioBuilder

class AudioCalibratorFactory(_Factory):
    '''Factory method for model object creation. Documented in parent class.

    '''
    def __init__(self):
        super(AudioCalibratorFactory, self).__init__()

factory = AudioCalibratorFactory()
factory.register_builder('paired audio recreation', CalibratorPairedAudioBuilder())