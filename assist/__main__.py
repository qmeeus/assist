import os
import click
import multiprocessing as mp
import shutil
from configparser import ConfigParser
from pathlib import Path


"""
TODO: config
 python -m assist
    -c 10-acquisition.cfg       --> parsed to ctx.obj["config"] by order of priority
    -c 20-train.cfg             --> idem
    -c recipe                   --> parse all files matching *.cfg in directory

TODO: add expdir as top level option and write config to file
 `ctx.obj["config"].write(expdir/f"{ctx.command}.cfg") `

Config sections
---------------
    - acquisition : eg. hidden_dim, num_layers, etc.
    - train : epochs, batch_size, datasections, etc.
    - test : batch_size, metric, datasections, etc.
    - coder : coder type, structure file, etc.
    - features : features type, paths to build database file, feature types, storage, etc.
    - cross_validation: number of splits, min examples per split, datasections, etc.
    - grid_search: path to parameter grid, datasections, etc.
"""


def read_config(ctx, param, paths):
    parser = ConfigParser()
    for path in sorted(paths):
        path = Path(path)
        if not Path(path).exists():
            raise click.BadParameter(f"{path} not found")
        # Let the parser manage the errors
        parser.read(paths)
    return parser

def init_logger(ctx, param, verbose):
    verbose = min(max(0, verbose), 3)
    os.environ["LOGLEVEL"] = {
        0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"
    }.get(verbose)
    from assist.tools import logger
    return logger


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", "logger", count=True, callback=init_logger)
@click.option("-c", "--config_file", "config", multiple=True, callback=read_config)
@click.option("--backend", default="local", type=click.Choice(["local", "mp", "condor"]))
@click.option("--cuda/--no-cuda", is_flag=True, default=True)
@click.option("--njobs", default=1, type=int)
@click.option("--overwrite", is_flag=True)
@click.pass_context
def cli(ctx, logger, config, backend, cuda, njobs, overwrite):
    if njobs < 0:
        njobs = mp.cpu_count()
    ctx.ensure_object(dict)
    ctx.obj["logger"] = logger
    ctx.obj["config"] = config
    ctx.obj["backend"] = backend
    ctx.obj["cuda"] = cuda
    ctx.obj["n_jobs"] = njobs
    ctx.obj["overwrite"] = overwrite
    click.echo("LOGLEVEL: " + os.environ["LOGLEVEL"])

@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--resume", is_flag=True)
@click.option("--retrain", is_flag=True)
@click.option("--clean", is_flag=True)
@click.pass_context
def cross_validation(ctx, expdir, recipe, resume=False, retrain=False, clean=False):

    from assist.scripts import prepare_cross_validation, run_cross_validation

    logger = ctx.obj["logger"]

    expdir, recipe = map(Path, (expdir, recipe))

    if (resume or retrain) and ctx.obj["overwrite"]:
        logger.warning("Setting overwrite flag to False to remain consistent")

    if expdir.exists() and ctx.obj["overwrite"]:
        logger.warning(f"{expdir} and its contents will be deleted")
        shutil.rmtree(expdir)
    elif expdir.exists() and not (resume or retrain):
        raise click.BadParameter(
            f"{expdir} exists and overwrite/resume/retrain flags not set")

    if not expdir.exists():
        prepare_cross_validation(expdir, recipe)

    queue = list(expdir.rglob("*blocks_exp*"))

    if resume:
        queue = list(filter(lambda exp: not((exp/"model").exists()), queue))

    if not queue:
        logger.error("Empty queue. Was resume flag set? Is the filesystem ok?")
        return

    logger.warning(f"{len(queue)} experiments in the queue.")

    run_cross_validation(
        expdir,
        queue,
        backend=ctx.obj["backend"],
        cuda=ctx.obj["cuda"],
        njobs=ctx.obj["n_jobs"],
        clean=clean
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--resume", is_flag=True)
@click.option("--no-eval", is_flag=True)
@click.pass_context
def train(ctx, expdir, recipe, resume, no_eval):

    from assist.scripts import prepare_train, run_train


    if ctx.obj["n_jobs"] > 1:
        raise click.BadParameter("For more than one job, use `train_many`")

    logger = ctx.obj["logger"]
    expdir, recipe = map(Path, (expdir, recipe))

    if resume and ctx.obj["overwrite"]:
        logger.info("Setting overwrite to False")
        ctx.obj["overwrite"] = False

    if expdir.exists() and ctx.obj["overwrite"]:
        logger.warning(f"{expdir} and its contents will be deleted")
        shutil.rmtree(expdir)
    elif expdir.exists() and not resume:
        raise click.BadOptionUsage("expdir", f"{expdir} exists and none of overwrite and resume flags are set")

    if not expdir.exists():
        prepare_train(expdir, recipe)

    run_train(
        expdir,
        backend=ctx.obj["backend"],
        cuda=ctx.obj["cuda"],
        njobs=ctx.obj["n_jobs"],
        do_eval=not(no_eval)
    )


@cli.command()
@click.argument("expdirs", nargs=-1, type=click.Path(exists=True))
@click.option("--no-eval", is_flag=True)
@click.pass_context
def train_many(ctx, expdirs, no_eval):
    # TODO: merge with train
    from assist.scripts import run_train

    run_train(
        list(map(Path, expdirs)),
        backend=ctx.obj["backend"],
        cuda=ctx.obj["cuda"],
        njobs=ctx.obj["n_jobs"],
        do_eval=not(no_eval)
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--clean", is_flag=True)
@click.pass_context
def evaluate(ctx, expdir, recipe, clean=False):

    from assist.scripts import run_evaluate

    run_evaluate(
        Path(expdir),
        backend=ctx.obj["backend"],
        cuda=ctx.obj["cuda"],
        clean=clean
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.pass_context
def prepare_dataset(ctx, expdir):

    from assist.tasks import dataset

    dataset.prepare_dataset(expdir)


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.pass_context
def prepare_database(ctx, expdir, recipe):

    from assist.scripts import run_prepare_database
    from assist.tools import logger

    expdir, recipe = map(Path, (expdir, recipe))

    if expdir.exists() and ctx.obj["overwrite"]:
        logger.warning(f"Deleting {expdir}")
        shutil.rmtree(expdir)
    elif expdir.exists():
        raise ValueError(f"{expdir} exists and flag overwrite not set")

    run_prepare_database(
        expdir,
        recipe,
        backend=ctx.obj["backend"],
        njobs=ctx.obj["n_jobs"],
        overwrite=ctx.obj["overwrite"]
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--resume", is_flag=True)
@click.option("--learning-curve", is_flag=True)
@click.pass_context
def gridsearch(ctx, expdir, recipe, no_data_prep, resume, learning_curve):

    from assist.scripts.gridsearch import prepare_gridsearch, gs_learning_curve

    logger = ctx.obj["logger"]

    expdir, recipe = map(Path, (expdir, recipe))

    if resume and ctx.obj["overwrite"]:
        logger.warning("Setting overwrite flag to False to remain consistent")

    if expdir.exists() and ctx.obj["overwrite"]:
        logger.warning(f"{expdir} and its contents will be deleted")
        shutil.rmtree(expdir)
    elif expdir.exists() and not resume:
        raise ValueError(f"{expdir} exists and overwrite/resume/no-data-prep flags not set")

    if not expdir.exists():
        prepare_gridsearch(expdir, recipe)

    gs_func = gs_learning_curve  #if learning_curve else gridsearch
    gs_func(expdir, recipe, cuda=ctx.obj["cuda"], n_jobs=ctx.obj["n_jobs"])


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.pass_context
def prepare_data(ctx, expdir, recipe):
    # backward compat TODO: remove
    from assist.scripts.prepare_dataprep import main

    main({
        "expdir": expdir,
        "recipe": recipe,
        "backend": ctx.obj["backend"],
        "cuda": ctx.obj["cuda"],
        "njobs": ctx.obj["n_jobs"],
        "overwrite": ctx.obj["overwrite"],
    })


if __name__ == "__main__":
    cli(obj={})
