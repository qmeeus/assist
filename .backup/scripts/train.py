import os
import sys
sys.path.append(os.getcwd())
import argparse
import random
from configparser import ConfigParser
import numpy as np
import re
from pathlib import Path
from assist.tasks import Structure, coder_factory
from assist.acquisition import model_factory
from assist.tools import logger, parse_line, read_config


def main(expdir, cuda):

    expdir = Path(expdir)

    #check if this experiment has been completed
    if (expdir/"model").exists():
        logger.warning(f"Found trained model in {expdir}.")
        return

    acquisitionconf = read_config(expdir/"acquisition.cfg")
    acquisitionconf.set("acquisition", "device", "cuda" if cuda else "cpu")

    coderconf = read_config(expdir/"coder.cfg")
    structure = Structure(os.path.join(expdir, 'structure.xml'))
    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)

    Model = model_factory(acquisitionconf.get('acquisition', 'name'))
    model = Model(acquisitionconf, coder, expdir)

    trainconf = dict(
        read_config(expdir/"train.cfg", default=Path(__file__).parent/"defaults/train.cfg")
        .items("train")
    )

    with open(expdir/"trainfeats") as trainfeats:
        features = {
            uttid: np.load(featsfile)
            for uttid, featsfile in map(parse_line, trainfeats.readlines())
        }

    with open(expdir/"traintasks") as traintasks:
        taskstrings = {
            uttid: task
            for uttid, task in map(parse_line, traintasks.readlines())
        }

    examples = {utt: (features[utt], taskstrings[utt]) for utt in taskstrings if utt in features}
    model.train(examples)
    model.save(expdir/'model')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    main(args.expdir, args.cuda)
