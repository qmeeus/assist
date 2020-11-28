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
from assist.tools import tools, logger


def load_hdf5(expdir, featconf, dataconf):
    import h5py

    speaker = expdir.name
    aggfunc = {
        "average": partial(np.mean, axis=0),
        "first": itemgetter(0)
    }.get(featconf.get("features", "agg"), None)

    with h5py.File(featconf.get('features', 'file'), "r") as h5f:
        keys = list(filter(lambda key: key.startswith(speaker), h5f.keys()))
        filenames = list(map(lambda name: Path(dataconf['features'], f"{name}.npy"), keys))
        for key, fn in zip(keys, filenames):
            array = h5f[key][()]
            if aggfunc is not None:
                array = aggfunc(array)
            np.save(fn, array)

    with open(Path(dataconf['features'], 'feats'), 'w') as featfile:
        featfile.writelines(map(lambda t: "{} {} \n".format(*t), zip(keys, filenames)))


def encode_feats(expdir, featconf, dataconf):
    from transformers import AutoTokenizer, AutoModel

    speaker = expdir.name

    model_string = featconf.get("features", "encoder")
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    encoder = AutoModel.from_pretrained(model_string)
    encoder.eval()

    with open(featconf.get("features", "file"), "r") as textfile:
        uttids, texts = map(list, zip(
            *map(tools.parse_line, filter(
                lambda line: line.startswith(speaker), 
                textfile.readlines()
            ))
        ))
        inputs = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True)
        input_lengths = inputs["attention_mask"].sum(-1)
        encoded, pooled = encoder(**inputs)
        if featconf.get("features", "agg") == "pooled":
            encoded = pooled
            aggfunc = None
        else:
            aggfunc = {
                "average": partial(np.mean, axis=0),
                "first": itemgetter(0)
            }.get(featconf.get("features", "agg"), None)

        filenames = list(map(lambda name: Path(dataconf['features'], f"{name}.npy"), uttids))
        for i, (uttid, fn) in enumerate(zip(uttids, filenames)):
            array = encoded[i, :input_lengths[i]].detach().numpy()
            if aggfunc is not None:
                array = aggfunc(array)
            np.save(fn, array)

    with open(Path(dataconf['features'], 'feats'), 'w') as featfile:
        featfile.writelines(map(lambda t: "{} {} \n".format(*t), zip(uttids, filenames)))


def compute_feats(expdir, featconf, dataconf):
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


def main(expdir):

    expdir = Path(expdir)
    featureconf_file = expdir/"features.cfg"
    dataconf_file = expdir/"database.cfg"

    dataconf = dict(tools.read_config(dataconf_file).items("database"))
    featconf = tools.read_config(featureconf_file)
    os.makedirs(dataconf['features'], exist_ok=True)

    processing_func = {
        "precomputed": load_hdf5,
        "text": encode_feats
    }.get(featconf.get("features", "name"), compute_feats)

    processing_func(expdir, featconf, dataconf)


if __name__ == "__main__":

    #create the arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=Path)
    main(parser.parse_args().expdir)
