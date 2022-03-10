'''Initialization upon import of `models`

The import generates:
 * Access to `factory`. This is the model object creation factory. It provides a uniform interface to all
models.

Written by: Anders Ohrn, March 2020

'''
from kinocilium.core._factory import _Factory
from kinocilium.core.models.conv_tasnet import ConvTasNetModelBuilder

class AudioModelFactory(_Factory):
    '''Factory method for model object creation. Documented in parent class.

    '''
    def __init__(self):
        super(AudioModelFactory, self).__init__()

factory = AudioModelFactory()
factory.register_builder('conv_tasnet', ConvTasNetModelBuilder())