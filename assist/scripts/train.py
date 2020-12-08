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

    features = FeatLoader(expdir/"trainfeats").to_dict()

    with open(expdir/"traintasks") as traintasks:
        taskstrings = {
            uttid: task
            for uttid, task in map(parse_line, traintasks.readlines())
        }

    examples = {
        utt: (features[utt], taskstrings[utt])
        for utt in taskstrings
        if utt in features
    }

    if not examples:
        raise ValueError("No training examples")

    model.train(examples)
    model.save(expdir/'model')

    # train_set, = model.prepare_inputs([x[0] for x in examples.values()])
    # probs = model.predict_proba(*model.prepare_inputs(train_set.features))
    # y_pred = (probs > .5).astype(int)
    # from assist.tasks import read_task
    # y_true = np.array([coder.encode(read_task(example[1])) for example in examples.values()])
    # from sklearn.metrics import classification_report
    # for line in classification_report(y_true, y_pred).split("\n"):
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
                    uttids, values = zip(*map(parse_line, f.readlines()))
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

