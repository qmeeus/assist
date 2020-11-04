'''@file mathmapper.py
contains the MATHMAPPER acoustic model'''

from acoustic_model import AcousticModel
import numpy

class MATHMAPPER(AcousticModel):
    '''a MATHMAPPER acoustic model

    a MATHMAPPER acoustic model copies the input to the output via a static function.
    Use identity if your features are posteriorgrams.
    '''

    def __init__(self, conf):
        '''
        AcousticModel constructor

        args:
            conf: the acoustic model configuration as a dictionary
        '''

        #super constructor
        super(MATHMAPPER, self).__init__(conf)
        scale = float(self.conf['scale'])
        self.fun = None
        name = self.conf['function']
        if name == 'exp':
            self.fun = lambda x: numpy.exp(x/scale)
        elif name == 'dummy':
            self.fun = lambda x: x/scale
        else:
            raise Exception('unknown acoustic mapper: %s' % name)


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

        return self.fun(features)

    @property
    def dim(self):
        '''the dimenionality of the encoded features'''

        return int(self.conf['dim'])
