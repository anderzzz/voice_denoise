'''Unit tests of audiodata classes

'''
import pytest

from kinocilium.core.audiodata import factory

data = factory.create('plain wav', path_to_folder='./data/data1')
for dd in data:
    print (dd)
