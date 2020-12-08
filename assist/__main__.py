import os
import click
import multiprocessing as mp
import shutil
from pathlib import Path


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", count=True)
@click.option("--backend", default="local", type=click.Choice(["local", "mp", "condor"]))
@click.option("--cuda/--no-cuda", is_flag=True, default=True)
@click.option("--njobs", default=mp.cpu_count(), type=int)
@click.option("--overwrite", is_flag=True)
@click.pass_context
def cli(ctx, verbose, backend, cuda, njobs, overwrite):
    verbose = min(max(0, verbose), 3)
    os.environ["LOGLEVEL"] = {
        0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"
    }.get(verbose)
    ctx.ensure_object(dict)
    ctx.obj["BACKEND"] = backend
    ctx.obj["CUDA"] = cuda
    ctx.obj["NJOBS"] = njobs
    ctx.obj["OVERWRITE"] = overwrite
    click.echo("LOGLEVEL: " + os.environ["LOGLEVEL"])


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--resume", is_flag=True)
@click.pass_context
def cross_validation(ctx, expdir, recipe, resume):

    from assist.scripts import prepare_cross_validation, run_cross_validation
    from assist.tools import logger

    expdir, recipe = map(Path, (expdir, recipe))

    if expdir.exists() and ctx.obj["OVERWRITE"]:
        logger.warning(f"{expdir} and its contents will be deleted")
        shutil.rmtree(expdir)
    elif expdir.exists() and not resume:
        raise ValueError(f"{expdir} exists and flags resume or overwrite not set")

    if not expdir.exists():
        prepare_cross_validation(expdir, recipe)

    queue = list(filter(
        lambda exp: not((exp/"model").exists()),
        expdir.rglob("*blocks_exp*")
    ))

    if not queue:
        raise ValueError("Empty queue. Was resume flag set? Is the filesystem ok?")

    run_cross_validation(
        expdir,
        queue,
        backend=ctx.obj["BACKEND"],
        cuda=ctx.obj["CUDA"],
        njobs=ctx.obj["NJOBS"]
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.option("--no-eval", is_flag=True)
@click.pass_context
def train(ctx, expdir, recipe, no_eval):

    from assist.scripts import prepare_train, run_train
    from assist.tools import logger

    if ctx.obj["NJOBS"] > 1:
        raise ValueError("For more than one job, use `train_many`")

    expdir, recipe = map(Path, (expdir, recipe))

    if expdir.exists() and ctx.obj["OVERWRITE"]:
        logger.warning(f"{expdir} and its contents will be deleted")
        shutil.rmtree(expdir)
    elif expdir.exists():
        raise ValueError(f"{expdir} exists and flag overwrite not set")

    if not expdir.exists():
        prepare_train(expdir, recipe)

    run_train(
        expdir,
        backend=ctx.obj["BACKEND"],
        cuda=ctx.obj["CUDA"],
        njobs=ctx.obj["NJOBS"],
        do_eval=not(no_eval)
    )


@cli.command()
@click.argument("expdirs", nargs=-1, type=click.Path(exists=True))
@click.option("--no-eval", is_flag=True)
@click.pass_context
def train_many(ctx, expdirs, no_eval):

    from assist.scripts import run_train

    run_train(
        list(map(Path, expdirs)),
        backend=ctx.obj["BACKEND"],
        cuda=ctx.obj["CUDA"],
        njobs=ctx.obj["NJOBS"],
        do_eval=not(no_eval)
    )


@cli.command()
@click.argument("expdir", type=click.Path())
@click.argument("recipe", type=click.Path(exists=True))
@click.pass_context
def evaluate(ctx, expdir, recipe):

    from assist.scripts import run_evaluate

    run_evaluate(Path(expdir), backend=ctx.obj["BACKEND"], cuda=ctx.obj["CUDA"])


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

    if expdir.exists() and ctx.obj["OVERWRITE"]:
        logger.warning(f"Deleting {expdir}")
        shutil.rmtree(expdir)
    elif expdir.exists():
        raise ValueError(f"{expdir} exists and flag overwrite not set")

    run_prepare_database(
        expdir,
        recipe,
        backend=ctx.obj["BACKEND"],
        njobs=ctx.obj["NJOBS"],
        overwrite=ctx.obj["OVERWRITE"]
    )



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
        "backend": ctx.obj["BACKEND"],
        "cuda": ctx.obj["CUDA"],
        "njobs": ctx.obj["NJOBS"],
        "overwrite": ctx.obj["OVERWRITE"],
    })


if __name__ == "__main__":
    cli(obj={})
