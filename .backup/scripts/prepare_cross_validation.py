import os
from pathlib import Path
import shutil
import argparse
import pickle
from configparser import ConfigParser
import random
import itertools
import numpy as np
import multiprocessing as mp
from assist.tasks import Structure, coder_factory, read_task
from assist.experiment import make_blocks
from assist.tools import tools, logger


def main(options):

    expdir = Path(options["expdir"])
    recipe = Path(options["recipe"])
    backend = options["backend"]
    cuda = options["cuda"]
    njobs = options["njobs"]
    overwrite = options["overwrite"]
    resume = options["resume"]

    if resume:
        logger.warning("Resume cross validation")
        overwrite = False
    if expdir.exists() and not overwrite and not resume:
        raise FileExistsError(expdir)
    elif expdir.exists() and overwrite:
        logger.warning(f"Remove expdir {expdir}")
        shutil.rmtree(expdir)

    queue = prepare_cross_validation(expdir, recipe)
    run_cross_validation(expdir, queue, backend=backend, njobs=njobs, cuda=cuda)


# def convert_key(key, speaker):
#     key = key.replace(speaker, "")
#     if key.startswith("_"):
#         key = key[1:]
#     return key


def run_cross_validation(expdir, queue, backend="local", njobs=12, cuda=False):

    if backend == "local":
        from assist.scripts import train_test

        if njobs > 1:
            with mp.Pool(njobs) as pool:
                pool.map(train_test.main, queue)
        else:
            for subexpdir in queue:
                train_test.main(subexpdir)

    elif backend == "condor":
        os.makedirs(expdir/"outputs", exist_ok=True)
        splits = list(map(list, np.array_split(queue, njobs)))
        queue_file = expdir/"condor_queue.txt"
        with open(queue_file, "w") as f:
            f.writelines([
                " ".join(line) + "\n" for line in splits
            ])
        device = "cuda" if cuda else "cpu"
        command = f"condor_submit assist/condor/run_many_{device}.job "
        if not cuda:
            command += f"ncpus={max(map(len, splits))} "
        submit_options = {
            "script": "train_test",
            "expdir": expdir,
            "queue_file": queue_file
        }
        command += " ".join([f"{key}={value}" for key, value in submit_options.items()])
        logger.info(tools.run_shell(command))

    else:
        raise NotImplementedError(f"backend={backend}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=Path, help='the experiments directory')
    parser.add_argument('recipe', type=Path, help='the recipe directory')
    parser.add_argument('--backend', default="local", choices=["local", "condor"])
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--njobs", default=12, type=int, help="Number of jobs (when computing == mp)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output directory")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume interrupted task")
    parser.add_argument("--random-seed", type=int, default=3105)
    main(vars(parser.parse_args()))
