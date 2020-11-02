'''@file dummy.py
contains the DUMMY acoustic model'''

from acoustic_model import AcousticModel

class DUMMY(AcousticModel):
    '''a DUMMY acoustic model

    a DUMMY acoustic model copies the input to the output, i.e. use it if your features are posteriorgrams.
    '''

    def __init__(self, conf):
        '''
        AcousticModel constructor

        args:
            conf: the acoustic model configuration as a dictionary
        '''

        #super constructor
        super(DUMMY, self).__init__(conf)


    def train(self, features):
        '''train the acoustic model

        args:
            features: a list of [length x dim] numpy arrays containing
                the features
        '''


    def save(self, savedir):
        '''save the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''

    def load(self, savedir):
        '''load the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''



    def __call__(self, features):
        '''
        use the acoustic model

        args:
            features: a length x featdim numpy arrays containing the
                features

        returns:
            a length x dim dimesional numpy array
        '''

        return features

    @property
    def dim(self):
        '''the dimenionality of the encoded features'''

        return int(self.conf['dim'])
