import numpy as np
import os
from pathlib import Path
import shutil
import subprocess

from assist.acquisition import model_factory
from assist.tasks import Structure, coder_factory
from assist.tools import FeatLoader, condor_submit, logger, mp_map, parse_line, read_config

from .evaluate import evaluate


def prepare_train(expdir, recipe):

    os.makedirs(expdir)

    for filename in ("acquisition.cfg", "coder.cfg", "train.cfg", "test.cfg", "structure.xml"):
        logger.debug(f"Copy {filename} from {recipe} to {expdir}")
        shutil.copy(recipe/filename, expdir/filename)

    dataconf = read_config(recipe/"database.cfg")

    for subset in ("train", "test"):
        prepare_subset(expdir, subset, dataconf)


def run_train(expdir, backend="local", cuda=False, do_eval=True, njobs=1):

    if backend == "local":
        if type(expdir) is not list:
            expdir = [expdir]
        N = len(expdir)
        mp_map(map_train, expdir, [cuda] * N, [do_eval] * N, njobs=njobs)
    elif backend == "condor":
        condor_submit(
            expdir,
            "train",
            [expdir],
            script_args="" if do_eval else "--no-eval",
            cuda=cuda
        )
    else:
        raise NotImplementedError(f"backend={backend}")


def map_train(args):
    train(*args)


def train(expdir, cuda=False, do_eval=True):
    logger.info(f"Train {expdir}")

    acquisitionconf = read_config(expdir/"acquisition.cfg")
    acquisitionconf.set("acquisition", "device", "cuda" if cuda else "cpu")

    coderconf = read_config(expdir/"coder.cfg")
    structure = Structure(expdir/'structure.xml')
    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)

    Model = model_factory(acquisitionconf.get('acquisition', 'name'))
    model = Model(acquisitionconf, coder, expdir)
    model.display(logger.info)

    trainfeats = FeatLoader(expdir/"trainfeats").to_dict()

    with open(expdir/"traintasks") as f:
        traintasks = dict(map(parse_line, f))

    train_set = {
        utt: (trainfeats[utt], traintasks[utt])
        for utt in traintasks
        if utt in trainfeats
    }

    if not train_set:
        raise ValueError("No training examples")

    test_feats = FeatLoader(expdir/"testfeats").to_dict()

    with open(expdir/"testtasks") as testtasks:
        test_tasks = dict(map(parse_line, testtasks))

    test_set = {
        utt: (test_feats[utt], test_tasks[utt])
        for utt in test_tasks
        if utt in test_feats
    }

    if (expdir/"model").exists():
        model.load(expdir/"model")

    model.train(train_set, test_set)
    model.save(expdir/'model')

    # from assist.tasks import read_task
    # from sklearn.metrics import classification_report, log_loss
    # from operator import itemgetter
    # from functools import partial

    # predictions = model.encode(model._decode(list(map(itemgetter(0), train_set.values()))))
    # target = model.encode(list(map(itemgetter(1), train_set.values())))

    # for line in classification_report(target, predictions).split("\n"):
    #     logger.info(line)

    if do_eval:
        evaluate(expdir, cuda=cuda)


def prepare_subset(expdir, subset, dataconf):

    conf = read_config(expdir/f"{subset}.cfg")

    logger.debug(f"Create {subset}feats and {subset}tasks files")
    with open(expdir/f"{subset}feats", "w") as feats, \
            open(expdir/f"{subset}tasks", "w") as tasks:
        for section in conf.get(subset, "datasections").split():
            featfile, taskfile = (
                Path(dataconf.get(section, key)) for key in ["features", "tasks"]
            )
            scpfile = str(featfile).replace(featfile.suffix, ".scp")
            for filepath, outfile in zip((scpfile, taskfile), (feats, tasks)):
                with open(filepath) as f:
                    uttids, values = zip(*map(parse_line, f))
                    if not(uttids[0].startswith(section)):
                        # Make sure that uttid has format $speaker_$uttid
                        uttids = list(map(f"{section}_{{}}".format, uttids))

                    outfile.writelines([
                        f"{uttid} {value}\n" for uttid, value in zip(uttids, values)
                    ])

    nfeats, ntasks = (
        subprocess.check_output(f"wc -l {expdir}/{filename}".split()).decode("utf-8").split()[0]
        for filename in (f"{subset}feats", f"{subset}tasks")
    )
    logger.info(f"Written {nfeats} features and {ntasks} tasks to {expdir} ({subset})")

