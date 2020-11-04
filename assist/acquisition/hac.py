'''@file hac.py
contains the hac function'''

import numpy as np

def hac(feats, delays, numkeep, censor=[]):
    '''computes the HAC vector

    args:
        feats: the input features as a TxF array
        delays: the HAC delays as a list of lists of integers
        numkeep: the number of events to keep at each timestep
        censor: index pairs that are set to zero; e.g. censor = [(0,0), (21,23)]

    returns:
        the HAC vector as a len(delays)*F**2 vector
    '''

    T, F = feats.shape

    if numkeep:
        sfeats = np.zeros(feats.shape)
        indy = np.argsort(-feats, 1)[:, :numkeep]
        indx = np.arange(T)[:, np.newaxis]
        sfeats[indx, indy] = feats[indx, indy]
    else:
        sfeats = feats

    hacs = []
    for d in delays:
        hacs.append([])
        for delay in d:
            if delay>0:
                padded = np.concatenate([np.zeros([delay, F]), sfeats])
                h = padded[:-delay].T.dot(padded[delay:])
                for c in censor:
                    h[c[0],c[1]] = 0
            else:
                h = sfeats.sum(0)
                for c in censor:
                    h[c[0]] = 0
            hacs[-1].append(h.reshape([-1]))

    hacs = [sum(h)/len(h) for h in hacs]

    return np.concatenate(hacs)

def ac(feats, delays, numkeep):
    '''computes the AC vector

    args:
        feats: the input features as a TxF array
        delays: the HAC delays as a list of integers
        numkeep: the number of events to keep at each timestep

    returns:
        the AC vector as a Txlen(delays)*F**2 array
    '''

    T, F = feats.shape

    if numkeep:
        sfeats = np.zeros(feats.shape)
        indy = np.argsort(-feats, 1)[:, :numkeep]
        indx = np.arange(T)[:, np.newaxis]
        sfeats[indx, indy] = feats[indx, indy]
    else:
        sfeats = feats

    acs = []
    for delay in delays:
        padded = np.concatenate([np.zeros([delay, F]), sfeats])
        acs.append(np.matmul(
            padded[:-delay, :, np.newaxis],
            padded[delay:, np.newaxis, :]).reshape([T, -1]))

    return np.concatenate(acs, 1)
