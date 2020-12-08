import os
import shutil
from configparser import ConfigParser
from pathlib import Path
from assist.tools import (
    logger,
    condor_submit,
    mp_map,
    parse_line,
    read_config,
    symlink
)

from .features import map_prepare_features


def run_prepare_database(expdir, recipe, backend="local", njobs=-1, overwrite=False):

    logger.debug(f"Create {expdir}")
    os.makedirs(expdir)

    dataconf = read_config(recipe/"database.cfg")
    shutil.copy(recipe/"features.cfg", expdir/"features.cfg")

    speakers = dataconf.sections()
    nspeakers = len(speakers)

    mp_map(
        map_prepare_spkrdir,
        [expdir] * nspeakers,
        speakers,
        [dict(dataconf[spkr].items()) for spkr in speakers],
        njobs=nspeakers
    )

    spkrdirs = [expdir/speaker for speaker in speakers]

    if backend == "condor":
        condor_submit(expdir, "prepare_dataset", spkrdirs)

    else:
        mp_map(map_prepare_features, spkrdirs, njobs=njobs)


def prepare_spkrdir(expdir, speaker, datafiles):
    """
    Prepare the database.cfg and features.cfg in expdir/speaker
    Parameters
    ----------
    expdir : pathlike
        The parent directory.
    speaker : str
        The identifier of the speaker.
    datafiles : dict
        the paths to files in the database relative to the speaker
    """

    logger.info(f"Prepare filesystem for speaker {speaker}")
    spkrdir = Path(expdir, speaker)
    os.makedirs(spkrdir, exist_ok=True)

    spkrconf = ConfigParser()
    spkrconf.read_dict({"database": datafiles})

    with open(spkrdir/"database.cfg", "w") as f:
        spkrconf.write(f)

    symlink(expdir/"features.cfg", spkrdir/"features.cfg")


def map_prepare_spkrdir(args):
    return prepare_spkrdir(*args)

