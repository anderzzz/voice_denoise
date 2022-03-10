'''General purpose object factory

Written by: Anders Ohrn, March 2022

'''
class _Factory(object):
    '''The guts of the general purpose object factory.

    Usage of this helps create uniform interfaces to different utilities despite that these may differ
    a great deal on the backend. This design is inspired by this website: https://realpython.com/factory-method-python/

    '''
    def __init__(self):
        self._builders = {}
        self.keys = self._builders.keys

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: An Object Builder instance

        '''
        self._builders[key] = builder

    def create(self, key, **kwargs):
        '''Method to create object via a registered object builder

        The `create` method is typically used where a relevant object is needed and the parameters for its creation
        are defined.

        Example:
             Importing factory and creating an object labelled `foo` with named arguments `bar` and `abc`::

                 from kinocilium.some_object_type import factory as object_type_factory
                 new_object = object_type_factory.create('foo', bar=3.14, abc=42)

        Args:
            key (str): The name of the object builder to use. Available values see attribute `keys`
            kwargs: Arbitrary length of named arguments to pass to the object builder

        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered object builder: {}'.format(key))
        return builder(**kwargs)