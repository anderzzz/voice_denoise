'''Bla bla

'''
from kinocilium.core.models.conv_tasnet import ConvTasNetModelBuilder

class AudioModelFactory(object):
    '''Interface to audio model factories.

    Typical usage involves the invocation of the `create` method, which returns a specific audio model

    '''
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: An Audio Model Builder instance

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

factory = AudioModelFactory()
factory.register_builder('conv_tasnet', ConvTasNetModelBuilder())