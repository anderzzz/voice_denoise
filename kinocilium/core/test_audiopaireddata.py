from kinocilium.core.audiodata import factory

data = factory.create('ms-snsd',
                      path_to_noisyspeech='/Users/andersohrn/Development/MS-SNSD/NoisySpeech_training',
                      path_to_cleanspeech='/Users/andersohrn/Development/MS-SNSD/CleanSpeech_training',
                      path_to_noise='/Users/andersohrn/Development/MS-SNSD/Noise_training',
                      read_metadata=False)
for dd in data:
    print (dd)