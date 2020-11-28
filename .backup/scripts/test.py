'''@file train_test.py
do training followed by testing
'''

import os
import sys
sys.path.append(os.getcwd())
import argparse
from configparser import ConfigParser
import numpy as np
from copy import deepcopy
from pathlib import Path
from assist.tasks import Structure, coder_factory, read_task, to_string
from assist.acquisition import model_factory
from assist.experiment import score, write_scores
from assist.tools import tools, logger



def main(expdir, cuda):

    expdir = Path(expdir)
    if (expdir/"f1").exists():
        logger.info(f"Results found at {expdir}")
        return

    logger.info(f"Evaluate {expdir}")

    acquisitionconf = tools.read_config(expdir/"acquisition.cfg")
    acquisitionconf.set("acquisition", "device", "cuda" if cuda else "cpu")
    coderconf = tools.read_config(expdir/"coder.cfg")
    structure = Structure(expdir/"structure.xml")

    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)
    Model = model_factory(acquisitionconf.get('acquisition', 'name'))
    model = Model(acquisitionconf, coder, expdir)
    logger.debug(f"Loading model at {expdir}/model")
    model.load(expdir/'model')

    with open(expdir/"testfeats") as testfeats:
        features = {
            line[0]: np.load(line[1])
            for line in map(tools.parse_line, testfeats.readlines())
        }

    with open(expdir/"testtasks") as testtasks:
        references = {
            key: read_task(value)
            for key, value in map(tools.parse_line, testtasks.readlines())
            if key in features
        }

    assert len(features) == len(references)

    #decode the test utterances
    feats = deepcopy(features)
    errors, nans, too_small = 0, 0, 0
    for uttid, feat in feats.items():
        remove = False
        # if feat.shape[0] < 5:
        #     too_small += 1
        #     remove = True
        if not np.isfinite(feat).all():
            nans += 1
            remove = True
        if remove:
            logger.debug(f"Removing {uttid}")
            errors += 1
            del features[uttid]
            del references[uttid]

    if errors > 0:
        logger.warning(
            f"{errors}/{len(feats)} utts removed ({too_small} too small and {nans} contained NaN)")

    decoded = model.decode(features)

    with open(expdir/"dectasks", "w") as dectasks_file:
        dectasks_file.writelines([
            f"{name}  {to_string(task)}\n"
            for name, task in decoded.items()
        ])

    metric_names = ["precision", "recal", "f1", "macro precision", "macro recal", "macro f1"]
    metrics, scores = score(decoded, references)

    for metric_name, metric in zip(metric_names, metrics):
        logger.info(f"{metric_name}: {metric:.4f}")
        with open(expdir/metric_name.replace(" ", ""), "w") as f:
            f.write(str(metric))

    write_scores(scores, expdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir')
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    main(args.expdir, args.cuda)

