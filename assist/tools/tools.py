'''@file tools.py
contains some usefull tools for the scripts'''

import os
import shutil
import warnings
import multiprocessing as mp

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


def run_shell(command):
    """
    Execute a command in the shell.
    Parameters
    ----------
    command : a valid UNIX command
    """
    logger.warning("Deprecated: use subprocess.check_output instead.")
    return os.popen(command).read().strip()


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


def condor_submit(expdir, script, queue, script_args="", cuda=False):
    """
    Start multiple condor jobs
    Parameters
    ----------
    expdir : pathlike
        The path where the condor files will be saved. It is passed to the script
    script : str
        The name of the script. It must be a valid assist command (see help)
    queue : list
        A list of the options to pass to the script
    """
    outdir = expdir/"outputs"
    os.makedirs(outdir, exist_ok=True)
    queue_file = outdir/"condor_queue.txt"
    with open(queue_file, "w") as f:
        f.writelines(queue)
    device = "cuda" if cuda else "cpu"
    command = f"condor_submit assist/condor/run_many_{device}.job "
    if not cuda:
        command += f"ncpus={max(map(len, queue))} "
    submit_options = {
        "script": script,
        "expdir": expdir,
        "script_args": script_args,
        "queue_file": queue_file
    }
    command += " ".join([f"{key}={value}" for key, value in submit_options.items()])
    logger.info(command)
    logger.warning(subprocess.check_output(command.split()).decode("utf-8"))
    logger.warning(f"Outputs saved to {outdir}")
