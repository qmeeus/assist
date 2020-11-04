'''@file dataprep.py
this script do the data preparation
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from configparser import ConfigParser
import numpy as np
from functools import partial
from operator import itemgetter
from pathlib import Path


def main(expdir):
    '''main function'''

    featureconf_file = os.path.join(expdir, 'features.cfg')
    dataconf_file = os.path.join(expdir, 'database.cfg')

    #read the data config file
    dataconf = ConfigParser()
    dataconf.read(dataconf_file)
    dataconf = dict(dataconf.items('database'))

    #read the features config file
    featconf = ConfigParser()
    featconf.read(featureconf_file)

    os.makedirs(dataconf['features'], exist_ok=True)

    if featconf.get('features', 'name') == "precomputed":
        speaker = Path(expdir).name
        aggfunc = (
            partial(np.mean, axis=0)
            if featconf.get('features', 'agg')
            else itemgetter(0)
        )
        import h5py
        with h5py.File(featconf.get('features', 'file')) as h5f:
            keys = list(filter(lambda key: key.startswith(speaker), h5f.keys()))
            filenames = list(map(lambda name: Path(dataconf['features'], f"{name}.npy"), keys))
            for key, fn in zip(keys, filenames):
                np.save(fn, aggfunc(h5f[key][()]))

        with open(Path(dataconf['features'], 'feats'), 'w') as featfile:
            featfile.writelines(map(lambda t: "{} {} \n".format(*t), zip(keys, filenames)))

        return

    from assist.features import feature_computer_factory
    from assist.features.read_audio import read_audio

    #create the feature computer
    feature_computer = feature_computer_factory.factory(
        featconf.get('features', 'name'))(featconf)

    #compute the features for all the audio files in the database and store them
    #on disk

    with open(os.path.join(dataconf['features'], 'feats'), 'w') as fid:
        for line in open(dataconf['audio']):
            splitline = line.strip().split(' ')
            name = splitline[0]
            print('computing features for %s' % name)
            wavfile = ' '.join(splitline[1:])
            rate, sig = read_audio(wavfile)
            if len(sig.shape) == 2:
                # feature computers assume mono
                sig = np.int16(np.mean(sig, axis=1))
            feats = feature_computer(sig, rate)
            filename = os.path.join(dataconf['features'], name + '.npy')
            fid.write(name + ' ' + filename + '\n')
            np.save(filename, feats)

if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    args = parser.parse_args()

    main(args.expdir)
