'''@file acoustic_model.py
contains the general acoustic model'''

from abc import ABCMeta, abstractmethod, abstractproperty

class AcousticModel(object):
    '''the acoustic model

    an acoustic model converts a sequence of feature vectors to a sequence of
    (distributions over) output labels'''

    __metaclass__ = ABCMeta

    def __init__(self, conf):
        '''
        AcousticModel constructor

        args:
            conf: the acoustic model configuration as a dictionary
        '''

        self.conf = dict(conf.items('acoustic'))

    @abstractmethod
    def train(self, features):
        '''train or adapt the acoustic model

        args:
            features: a list of [length x featdim] numpy arrays containing the
                features
        '''

    @abstractmethod
    def save(self, savedir):
        '''save the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''

    @abstractmethod
    def load(self, savedir):
        '''save the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''

    @abstractmethod
    def __call__(self, features):
        '''
        use the acoustic model

        args:
            features: a length x featdim numpy arrays containing the
                features

        returns:
            a dim dimesional numpy array
        '''

    @abstractproperty
    def dim(self):
        '''the dimenionality of the encoded features'''
