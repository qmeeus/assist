import multiprocessing as mp
import numpy as np
import os
import shutil
import subprocess
import warnings

from pathlib import Path
from configparser import ConfigParser

from .logger import logger


__all__ = [
    "condor_submit",
    "default_conf",
    "mp_map",
    "parse_line",
    "read_config",
    "run_shell",
    "safecopy",
    "symlink",
    "writefile",
]


def mp_map(func, *args, njobs=-1):
    """
    Execute a function in multiprocessing mode.
    Parameters
    ----------
    func : callable
        The function to be executed. It must be pickleable, take `len(args)`
        number of arguments and accept only positional arguments.
    *args: lists
        The arguments to pass to the function. Each argument is a list and all lists
        should be of equal length (although not enforced). If `njobs == 1`, the arguments
        can be of any type since they are passed to the function as is.
    njobs: int
        The number of jobs. Set to 1 to disable multiprocessing. When -1, it defaults to the
        number of CPUs. The default is -1.
    """
    if njobs == -1:
        njobs = mp.cpu_count()

    if njobs <= 1:
        return [func(opts) for opts in zip(*args)]

    with mp.Pool(njobs) as pool:
        return pool.map(func, zip(*args))


def condor_submit(expdir, command, queue, command_options="", cuda=False, njobs=1, dry_run=False):
    """
    Start multiple condor jobs
    Parameters
    ----------
    expdir : pathlike
        The path where the condor files will be saved. It is passed to the script
    command : str
        The name of the command. It must be a valid assist command (see `python -m assist -h`)
    queue : list
        A list that contains the variable elements of the command (e.g. expdirs when run_train_many)
    """
    outdir = expdir/"outputs"
    os.makedirs(outdir, exist_ok=True)
    queue_file = outdir/"condor_queue.txt"

    if njobs > 1 and command != "train-many":
        raise NotImplementedError("Multiple jobs only allowed for train-many command")
    elif njobs > 1:
        tasks = np.array_split(queue, len(queue) // njobs)
        tasklen = list(map(len, tasks))

        logger.info(
            f"{len(queue)} tasks allocated to {len(tasks)} jobs "
            f"(between {min(tasklen)} and {max(tasklen)} tasks per job)"
        )

        queue = [" ".join(map(str, jobs)) for jobs in tasks]
    else:
        logger.info(f"Submit {len(queue)} jobs to condor")

    queue = [f"{item} {command_options}" for item in queue]
    
    with open(queue_file, "w") as f:
        f.writelines([f"{item} {command_options}\n" for item in queue])

    device = "cuda" if cuda else "cpu"
    submit_command = f"condor_submit assist/condor/run_many_{device}.job "

    verbose = {10: "-vvv", 20: "-vv", 30: "-v"}.get(logger.getEffectiveLevel(), "")

    submit_options = {
        "njobs": njobs,
        "verbose": verbose,
        "command": command,
        "outdir": outdir,
        "queue_file": queue_file
    }
    
    submit_command += " ".join([f"{key}={value}" for key, value in submit_options.items()])
    run_shell(submit_command)
    logger.warning(f"Outputs saved to {outdir}")



def run_shell(command):
    """
    Execute a command in the shell.
    Parameters
    ----------
    command : Union[str, List[str]]
        A valid UNIX command, either as a string or as a list of strings. Note that if the command
        arguments must be protected (eg spaces), it is safer to pass the command as a list.
    """
    if not isinstance(command, list):
        command = command.split()
    logger.warning(" ".join(command))
    output = subprocess.check_output(command, shell=True).decode("utf-8")
    for line in output.split("\n"):
        logger.warning(line)


def isin(items):
    """
    Creates a function that checks whether an item is in a list of items
    Parameters
    ----------
    items : List[Any]
        The list against which values are checked
    Returns : Callable
    """
    def _isin(item):
        return item in items
    return _isin


def parse_line(line, sep=" "):
    line = line.strip()
    return line.split(sep, maxsplit=1)


def read_config(path, default=None):
    path = Path(path)
    if not(path.exists()) and (default is None or not Path(default).exists()):
        raise FileNotFoundError(f"{path} was not found.")
    parser = ConfigParser()
    parser.read(path)
    if default is not None and Path(default).exists():
        default_conf(parser, default)
    return parser


def symlink(source, link_name):
    '''
    create a symlink, if target exists remove

    args:
        source: the file the link will be pointing to
        link_name: the path to the link file
    '''

    source, link_name = map(Path, (source, link_name))

    if link_name.exists() or link_name.is_symlink():
        os.remove(link_name)

    os.symlink(source.resolve(), link_name)


def safecopy(src, dst):
    '''only copy src to dest if dst does not exits'''

    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def writefile(filename, strings):
    '''write a dictionary of strings to a file'''

    if not os.path.exists(filename):
        with open(filename, 'w') as fid:
            for name in strings:
                fid.write('%s %s\n' % (name, strings[name]))


def default_conf(conf, default_path):
    '''put the defaults in the configuration if it is not defined

    args:
        conf: the conf as a ConfigParser object
        default_path: the path to the default location
    '''

    #read the default conf
    default = ConfigParser()
    default.read(default_path)

    #write the default values that are not in the config to the config
    for section in default.sections():
        if not conf.has_section(section):
            conf.add_section(section)
        for option in default.options(section):
            if not conf.has_option(section, option):
                conf.set(section, option, default.get(section, option))
