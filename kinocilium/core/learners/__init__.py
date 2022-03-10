'''Initialization upon import of `learners`

The import generates:
 * Access to `factory`. This is the learner object creation factory. It provides a uniform interface to all
learners.

Written by: Anders Ohrn, March 2020

'''
from kinocilium.core._factory import _Factory
#from kinocilium.core.learners.FOOBAR import SOMETHING

class AudioLearnerFactory(_Factory):
    '''Factory method for model object creation. Documented in parent class.

    '''
    def __init__(self):
        super(AudioLearnerFactory, self).__init__()

factory = AudioLearnerFactory()
#factory.register_builder('FOOBAR', None)