import itertools
import numpy as np
import os
import pickle
import random
import shutil

from pathlib import Path

from assist.experiment import make_blocks
from assist.tasks import Structure, coder_factory, read_task
from assist.tools import (
    condor_submit,
    logger,
    mp_map,
    parse_line,
    read_config,
    symlink,
    writefile
)

from .train import map_train


def prepare_cross_validation(expdir, recipe):

    os.makedirs(expdir, exist_ok=True)
    for filename in ("acquisition.cfg", "coder.cfg", "structure.xml"):
        logger.debug(f"Copy {filename} from {recipe} to {expdir}")
        shutil.copy(recipe/filename, expdir/filename)

    expconf = dict(read_config(
        recipe/"cross_validation.cfg",
        default=Path(__file__).parent/"defaults/cross_validation.cfg"
    ).items("cross_validation"))

    random_seed = int(expconf.get("random_seed", 3105))
    logger.debug(f"Setting random seed to {random_seed}")
    random.seed(random_seed)

    dataconf = read_config(recipe/"database.cfg")
    coderconf = read_config(expdir/"coder.cfg")

    structure = Structure(expdir/'structure.xml')
    Coder = coder_factory(coderconf.get("coder", "name"))
    coder = Coder(structure, coderconf)

    option_list = [
        dict(
            expdir=expdir,
            speaker=speaker,
            coder=coder,
            dataconf=dataconf,
            expconf=expconf
        ) for speaker in dataconf.sections()
    ]

    for opts in option_list:
        map_prepare_filesystem(opts)


def run_cross_validation(expdir, queue, backend="local", njobs=12, cuda=False):
    if backend == "local":
        mp_map(map_train, queue, [cuda] * len(queue), njobs=njobs)
    else:
        condor_submit(expdir, "train", queue, cuda=cuda)


def map_prepare_filesystem(options):
    return prepare_filesystem(**options)


def prepare_filesystem(expdir, speaker, coder, dataconf, expconf):

    speaker_dir = expdir/speaker
    os.makedirs(speaker_dir, exist_ok=True)

    feature_file = Path(dataconf.get(speaker, 'features'))
    with open(str(feature_file).replace(feature_file.suffix, ".scp")) as featfile:
        features = dict(map(parse_line, featfile.readlines()))

    with open(dataconf.get(speaker, "tasks")) as taskfile:
        task_strings = {
            f"{speaker}_{uttid}": task for uttid, task in map(parse_line, taskfile.readlines())
        }
        for uttid in list(task_strings):
            if uttid not in features:
                logger.warning(f"Missing utterance speaker {speaker}: {uttid}")
                del task_strings[uttid]

        tasks = [coder.encode(read_task(task)) for task in task_strings.values()]

    assert tasks, "An error occured: no tasks"

    tasks = np.array(tasks)
    blocks_path = speaker_dir/"blocks.pkl"
    if blocks_path.exists():
        with open(blocks_path, "rb") as blockfile:
            blocks = pickle.load(blockfile)
    else:
        try:
            blocks = make_blocks(tasks, expconf, feature_file.parent)
        except Exception as err:
            logger.error(f"Error with speaker {speaker}: {err}")
            raise err
            return []
        with open(blocks_path, "wb") as blockfile:
            pickle.dump(blocks, blockfile)

    num_exp = int(expconf["numexp"])

    train_ids, test_ids = [], []
    for block_id in range(len(blocks) - 1):
        train_ids.append([])
        test_ids.append([])
        for exp_id in range(num_exp):
            train_ids[-1].append(list(itertools.chain.from_iterable(
                random.sample(blocks, block_id + 1)
            )))
            test_ids[-1].append(
                [i for i in range(len(tasks)) if i not in train_ids[-1][-1]]
            )

    if not(train_ids and test_ids):
        logger.error(f"Error with speaker {speaker}: no utterances")
        return []

    uttids = list(task_strings)
    block_id = int(expconf['startblocks']) - 1
    while True:

        dirname = f"{block_id + 1}blocks_exp"
        num_exp = int(expconf['numexp'])
        for exp_id in range(num_exp):

            subexpdir = expdir/speaker/(dirname + str(exp_id))
            logger.debug(f"Experiment {subexpdir.name}")

            if (subexpdir/"f1").exists():
                logger.info(f"Skipping {subexpdir}")
                continue

            os.makedirs(subexpdir, exist_ok=True)

            for filename in ("acquisition.cfg", "coder.cfg", "structure.xml"):
                symlink(expdir/filename, subexpdir/filename)

            if not (subexpdir/"trainfeats").exists():
                for subset, ids in [("train", train_ids), ("test", test_ids)]:
                    utts = [uttids[idx] for idx in ids[block_id][exp_id] if idx < len(uttids)]
                    if len(utts) != len(ids[block_id][exp_id]):
                        num_lost = len(ids[block_id][exp_id]) - len(utts)
                        logger.warning(f"Lost {num_lost} {subset} utterances")
                    logger.debug(f"Number of {subset} examples: {len(utts):,}")

                    writefile(subexpdir/f"{subset}feats", {utt: features[utt] for utt in utts})
                    writefile(subexpdir/f"{subset}tasks", {utt: task_strings[utt] for utt in utts})

        next_block_id = (block_id + 1) * int(expconf['scale']) + int(expconf['increment']) - 1
        next_block_id = min(next_block_id, len(blocks) - 2)
        if block_id == next_block_id:
            break
        else:
            block_id = next_block_id

