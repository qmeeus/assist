import argparse
import os
import shutil
import subprocess
from pathlib import Path
from configparser import ConfigParser
from assist.scripts import train
from assist.tools import logger, read_config


def main(expdir, recipe, backend, cuda):

    expdir, recipe = map(Path, (expdir, recipe))

    if expdir.exists():
        if input(f"{expdir} exists. Remove? ").strip().lower() == "y":
            logger.info(f"Removing {expdir}")
            shutil.rmtree(expdir)

    if not expdir.exists():
        prepare_traindir(expdir, recipe)

    run(expdir, backend, cuda)


def prepare_traindir(expdir, recipe):

    os.makedirs(expdir)

    for filename in ["acquisition.cfg", "coder.cfg", "train.cfg", "structure.xml"]:
        logger.debug(f"Copy {filename} from {recipe} to {expdir}")
        shutil.copy(recipe/filename, expdir)

    trainconf = read_config(expdir/"train.cfg")
    dataconf = read_config(recipe/"database.cfg")

    if "fluent" in str(expdir).lower():
        convert_uttid = lambda spkr, uttid: (spkr if not uttid.startswith(spkr) else "") + uttid
    else:
        convert_uttid = lambda spkr, uttid: (spkr + "_" if not uttid.startswith(spkr) else "") + uttid

    logger.debug("Create trainfeats and traintasks files")
    with open(expdir/"trainfeats", "w") as feats, open(expdir/"traintasks", "w") as tasks:
        for section in trainconf.get("train", "datasections").split():
            with open(Path(dataconf.get(section, "features"))/"feats") as f:
                feats.writelines([
                    convert_uttid(section, line) for line in f.readlines()
                ])
            with open(Path(dataconf.get(section, "tasks"))) as f:
                tasks.writelines([
                    convert_uttid(section, line) for line in f.readlines()
                ])

    nfeats, ntasks = (
        subprocess.check_output(f"wc -l {expdir}/{filename}".split())
        .decode("utf-8").split()[0] for filename in ("trainfeats", "traintasks")
    )
    logger.info(f"Written {nfeats} features and {ntasks} tasks to {expdir}")

def run(expdir, backend, cuda):
    if backend == 'condor':
        os.makedirs(expdir/"outputs", exist_ok=True)
        jobfile = "assist/condor/run_script{}.job".format("_GPU" if cuda else "")
        in_queue = os.popen(
            'if condor_q -nobatch -wide | grep -q %s; '
            'then echo true; else echo false; fi' %
            expdir).read().strip() == 'true'

        if not in_queue:
            condor_submit = f"condor_submit expdir={expdir} script=train {jobfile}"
            logger.warning(condor_submit)
            logger.warning(subprocess.check_output(condor_submit.split()).decode("utf-8"))
    else:
        logger.warning("Local training started")
        train.main(expdir, cuda)

if __name__ == '__main__':

    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', help='the experiments directory')
    parser.add_argument('recipe', help='the recipe directory')
    parser.add_argument('--backend', choices=["condor", "local"], help='the kind of backend you want to do')
    parser.add_argument('--cuda', action="store_true", default=False, help='run job on gpu')
    args = parser.parse_args()
    main(args.expdir, args.recipe, args.backend, args.cuda)

