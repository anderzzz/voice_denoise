'''Bla bla

'''
from kinocilium.core.data_getters.audio_plain import AudioPlainWAVDataBuilder
from kinocilium.core.data_getters.mssnsd_noisyspeech import AudioMSSNSDDataBuilder

class AudioDataFactory(object):
    '''Interface to audio data factories.

    Typical usage involves the invocation of the `create` method, which returns a specific audio dataset

    '''
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: An Audio Data Builder instance

        '''
        self._builders[key] = builder

    @property
    def keys(self):
        return self._builders.keys()

    def create(self, key, **kwargs):
        '''Method to create audio data set through uniform interface

        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered data builder: {}'.format(key))
        return builder(**kwargs)

factory = AudioDataFactory()
factory.register_builder('plain wav', AudioPlainWAVDataBuilder())
factory.register_builder('ms-snsd', AudioMSSNSDDataBuilder())