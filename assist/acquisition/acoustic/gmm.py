'''@file gmm.py
contains the GMM acoustic model'''

import os
import cPickle as pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from acoustic_model import AcousticModel

class GMM(AcousticModel):
    '''a GMM acoustic model

    a GMM converts the features into a sequence of distributions over gaussians
    '''

    def __init__(self, conf):
        '''
        AcousticModel constructor

        args:
            conf: the acoustic model configuration as a dictionary
        '''

        #super constructor
        super(GMM, self).__init__(conf)

        #create the GMM
        self.gmm = GaussianMixture(
            n_components=int(self.conf['components']),
            covariance_type=self.conf['covariance_type'],
            max_iter=int(self.conf['max_iter'])
        )

    def train(self, features):
        '''train the acoustic model

        args:
            features: a list of [length x dim] numpy arrays containing
                the features
        '''

        #concatenate all the features
        all_features = np.concatenate(features)
        self.gmm.fit(all_features)

    def save(self, savedir):
        '''save the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''
        location = os.path.join(savedir, 'gmm')

        if not os.path.isdir(location):
            os.makedirs(location)

        with open(os.path.join(location, 'gmm.pkl'), 'wb') as fid:
            pickle.dump(self.gmm, fid)

    def load(self, savedir):
        '''save the acoustic model

        args:
            savedir: the storage directory for acoustic models
        '''

        gmmfile = os.path.join(savedir, 'gmm', 'gmm.pkl')

        if not os.path.isfile(gmmfile):
            raise Exception('cannot find %s' % gmmfile)

        with open(gmmfile, 'rb') as fid:
            self.gmm = pickle.load(fid)


    def __call__(self, features):
        '''
        use the acoustic model

        args:
            features: a length x featdim numpy arrays containing the
                features

        returns:
            a length x dim dimesional numpy array
        '''

        if features.shape[0]<5:
            return np.zeros((0,self.dim),dtype=float)
        return self.gmm.predict_proba(features)

    @property
    def dim(self):
        '''the dimenionality of the encoded features'''

        return int(self.conf['components'])
