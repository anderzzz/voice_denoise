'''Initialization upon import of `data_getters`

The import generates:
 * Access to `factory`. This is the dataset getter object creation factory. It provides a uniform interface to all
data getters.

Written by: Anders Ohrn, March 2020

'''
from kinocilium.core._factory import _Factory
from kinocilium.core.data_getters.audio_plain import AudioPlainWAVDataBuilder
from kinocilium.core.data_getters.mssnsd_noisyspeech import AudioMSSNSDDataBuilder
from kinocilium.core.data_getters.audiominst import AudioMINSTDataBuilder

class AudioDataFactory(_Factory):
    '''Audio Data Factory. Documented in parent class

    '''
    def __init__(self):
        super(AudioDataFactory, self).__init__()

factory = AudioDataFactory()
factory.register_builder('plain wav', AudioPlainWAVDataBuilder())
factory.register_builder('ms-snsd', AudioMSSNSDDataBuilder())
factory.register_builder('audio-minst', AudioMINSTDataBuilder())