'''@file feature_computer.py
contains the FeatureComputer class'''

import os
from functools import partial
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.ndimage.filters import maximum_filter
from assist.tools.tools import default_conf
import math
from . import delta
import pickle

# need to be modified for on-line VAD
from assist.features import sigproc
from assist.features import base

def povey(N):
    return np.power(0.5 - 0.5 * np.cos( (2.0*np.pi/float(N))*np.arange(N) ), 0.85).astype(np.float32)

class FeatureComputer(object):
    '''A featurecomputer is used to compute features'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, rate):
        '''
        FeatureComputer constructor

        Args:
            conf: the feature configuration
        '''

        #default conf file
        default = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'defaults',
            type(self).__name__.lower() + '.cfg')

        #apply the defaults
        if os.path.exists(default):
            default_conf(conf, default)

        self.conf = dict(conf.items('features'))
        self.rate = rate
        self.CHUNK = int(float(self.conf['winstep'])*self.rate) # chunk size, typically 160
        nSamPerWin = int(float(self.conf['winlen'])*self.rate)
        # override nfft from config
        self.nfft = 2**math.ceil(math.log2(nSamPerWin))
        #self.win = np.concatenate((np.hamming(nSamPerWin), np.zeros(self.nfft-nSamPerWin)))
        #self.win = np.concatenate((povey(nSamPerWin), np.zeros(self.nfft-nSamPerWin)))
        # new samples placed at the end of the window => reduce delay 
        self.win = np.concatenate((np.zeros(self.nfft-nSamPerWin, dtype=np.float32), povey(nSamPerWin)))
        self.nChunkPerWin = int(math.ceil(nSamPerWin/self.CHUNK)) # needed ?
        self.x = np.zeros(self.nfft, dtype=np.float32)
        # for preemphesis
        self.lastsample = 0
        self.pre = float(self.conf['preemph'])
        # build feature vector
        self.include_energy = self.conf['include_energy'] == 'True'
        try:
            self.mvn = np.loadtxt(self.conf['mvn'])
            #self.mvn = np.fromfile(self.conf['mvn'])
            # with open(self.conf['mvn'], 'rb') as fin :
            #     self.mvn = pickle.load(fin)
        except:
            print("Mean/variance normalzation switched off.")
            self.mvn = None

    def set_delta(self):
        if self.conf['dynamic'] == 'nodelta':
            self.delta = lambda x: x
        elif self.conf['dynamic'] == 'delta':
            self.op1 = delta.Delta1(self.basedim)
            self.delta = lambda x: self.op1(x)
        elif self.conf['dynamic'] == 'ddelta':
            self.op1 = delta.Delta1(self.basedim)
            self.op2 = delta.Delta1(2*self.basedim)
            self.delta = lambda x: np.delete(self.op2(self.op1(x)), range(x.size,2*x.size) ) # ok, not efficient
        else:
            raise NameError("Delta operation %s undefinded " % (self.conf['dynamic']))

    def __call__(self, sig):
        '''
        compute the features

        Args:
            sig: a numpy array with a sample chunk of length "frame shift"

        Returns:
            the features as a numpy array of dimension feature dim
        '''

        # # pre-emphasis
        # pre = np.insert(sig[1:]-self.pre*sig[0:-1], 0, sig[0]-self.pre*self.lastsample)
        # self.lastsample = sig[-1]
        #
        # # windowing
        # self.x[self.CHUNK:self.nfft] = self.x[0:self.nfft-self.CHUNK]
        # self.x[0:self.CHUNK] = pre
        NewStart = self.nfft - self.CHUNK
        self.x[0:NewStart] = self.x[self.CHUNK:self.nfft] # shift old samples
        self.x[NewStart] = sig[0]-self.pre*self.lastsample
        self.x[NewStart+1:] = sig[1:]-self.pre*sig[0:-1]
        self.lastsample = sig[-1]

        #compute the features
        feats, energy = self.comp_feat(self.x * self.win)

        if self.include_energy:
            feats = np.append(feats, np.log(energy))
        feats = self.delta(feats)

        if self.mvn is not None:
            feats = self.mvn[1,:] * (feats - self.mvn[0,:])

        #apply vad
        if self.conf['vad'] == 'True':
            raise NameError('VAD not implemented yet')
            speechframes = vad(sig, self.rate, float(self.conf['winlen']),
                               float(self.conf['winstep']))
            feats = feats[speechframes, :]

        return feats

    @abstractmethod
    def comp_feat(self, sig):
        '''
        compute the features

        Args:
            sig: the audio signal as a 1-D numpy array
            rate: the sampling rate

        Returns:
            the features as a [seq_length x feature_dim] numpy array
        '''

    @abstractmethod
    def get_dim(self):
        '''the feature dimemsion'''


def vad(sig, rate, winlen, winstep):
    '''do voice activity detection

    args:
        sig: the input signal as a numpy array
        rate: the sampling rate
        winlen: the window length
        winstep: the window step

    Returns:
        a numpy array of indices containing speech frames
    '''

    #apply preemphasis
    sig = sigproc.preemphasis(sig, 0.97)

    #do windowing windowing
    frames = sigproc.framesig(sig, winlen*rate, winstep*rate)

    #compute the squared frames and center them around zero mean
    sqframes = np.square(frames)
    sqframes = sqframes - sqframes.mean(1, keepdims=True)

    #compute the cross correlation between the frames and their square
    corr = np.array(map(partial(np.correlate, mode='same'), frames, sqframes))

    #compute the mel power spectrum of the correlated signal
    corrfft = np.fft.rfft(corr, 512)
    fb = base.get_filterbanks(26, 512, rate, 0, rate/2)
    E = np.absolute(np.square(corrfft).dot(fb.T))

    #do noise sniffing at the front and the back and select the lowest energy
    Efront = E[:20, :].mean(0)
    Eback = E[-20:, :].mean(0)
    if Efront.sum() < Eback.sum():
        Enoise = Efront
    else:
        Enoise = Eback

    #at every interval compute the mean ratio between the maximal energy in that
    #interval and the noise energy
    width = 12

    #apply max pooling to the energy
    Emax = maximum_filter(E, size=[width, 1], mode='constant')

    #compute the ratio between the smoothed energy and the noise energy
    ratio = np.log((Emax/Enoise).mean(axis=1))
    ratio = ratio/np.max(ratio)

    speechframes = np.where(ratio > 0.2)[0]

    return speechframes
