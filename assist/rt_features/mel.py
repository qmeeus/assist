'''@file mel.py
contains the mel feature computer, which supports mfcc of fbank (by seting numcep<=0)'''

import numpy as np
from scipy.fftpack import dct
from assist.features.base import lifter
from assist.rt_features.base import get_filterbanks
from . import feature_computer

class Mel(feature_computer.FeatureComputer):
    '''the feature computer class to compute mfcc features'''

    def __init__(self, conf, rate):
        super().__init__(conf, rate)
        self.nmfcc = int(self.conf['numcep'])
        self.basedim = self.nmfcc # mfcc
        if self.basedim<=0: # fbank
            self.basedim = int(self.conf['nfilt'])
        if self.include_energy:
            self.basedim += 1
        self.dim = self.basedim
        if self.conf['dynamic'] == 'delta':
            self.dim *= 2
        elif self.conf['dynamic'] == 'ddelta':
            self.dim *= 3
        highfreq = int(self.conf['highfreq'])
        if highfreq <= 0:
            highfreq += self.rate / 2
        self.filterbank = get_filterbanks(nfilt=int(self.conf['nfilt']), nfft=int(self.nfft),
            samplerate=self.rate, lowfreq=int(self.conf['lowfreq']), highfreq=highfreq).T
        self.eps = 1e-5
        self.lifter = float(self.conf['ceplifter'])
        super().set_delta()

    def comp_feat(self, sig):
        complex_spec = np.fft.rfft(sig, len(sig))
        pspec = (np.square(complex_spec.real) + np.square(complex_spec.imag)).astype(np.float32)
        energy = np.sum(pspec + self.eps)
        melspec = np.dot(pspec, self.filterbank)
        features = np.log(np.where(melspec < self.eps, self.eps, melspec))
        if self.nmfcc > 0:
            features = dct(features, type=2, norm='ortho')[:self.nmfcc]
            features = np.squeeze(lifter(np.expand_dims(features,0), self.lifter))

        return features, energy

    def get_dim(self):
        '''the feature dimemsion'''

        return self.dim

class OldFbank(feature_computer.FeatureComputer):
    '''the feature computer class to compute fbank features'''

    def __init__(self, conf, rate):
        super().__init__(self, conf, rate)
        highfreq = int(self.conf['highfreq'])
        if highfreq < 0:
            highfreq = self.rate / 2
        self.filterbank = get_filterbanks(int(conf['nfilt']), int(conf['nfft']), self.rate,
                                          int(conf['lowfreq']), highfreq)

    def comp_feat(self, sig):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

        feat, energy = base.logfbank(sig, rate, self.conf)

        if self.conf['include_energy'] == 'True':
            feat = np.append(feat, energy[:, np.newaxis], 1)

        if self.conf['dynamic'] == 'delta':
            feat = base.delta(feat)
        elif self.conf['dynamic'] == 'ddelta':
            feat = base.ddelta(feat)
        elif self.conf['dynamic'] != 'nodelta':
            raise Exception('unknown dynamic type')

        #mean and variance normalize the features
        if self.conf['mvn'] == 'True':
            feat = (feat - feat.mean(0))/feat.std(0)

        return feat
