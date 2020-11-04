import click
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess


@click.group()
@click.option("--format", default="png", type=click.Choice(["png", "eps"]))
@click.option("--dpi", default=300, type=int)
@click.option("--overwrite", is_flag=True)
@click.option("--display", is_flag=True)
@click.pass_context
def cli(ctx, format, dpi, overwrite, display):
    ctx.ensure_object(dict)
    ctx.obj["FORMAT"] = format
    ctx.obj["DPI"] = dpi
    ctx.obj["OVERWRITE"] = overwrite
    ctx.obj["DISPLAY"] = display


@cli.command()
@click.argument("metrics", nargs=-1)
@click.option("--expdir", required=True, type=click.Path())
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.pass_context
def learning_curve(ctx, metrics, expdir, plot_speakers):
    fmt, dpi, display = ctx.obj["FORMAT"], ctx.obj["DPI"], ctx.obj["DISPLAY"]
    expdir = validate_dir(expdir)
    savedir = expdir / "figures"
    if savedir.exists() and not ctx.obj["OVERWRITE"]:
        raise ValueError(f"{savedir} exists already. Add --overwrite to overwrite.")
    os.makedirs(savedir, exist_ok=True)

    results = load_results(expdir, metrics)
    speakers = sorted(results.index.levels[0])

    if plot_speakers:
        for speaker in speakers:
            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), metrics
            )

            ax.set_title(f"Learning curve {expdir.name} {speaker}")
            plt.legend()
            plt.savefig(savedir / f"curve_{speaker}.{fmt}", dpi=dpi)

    for metric in metrics:
        ax = None

        for speaker in speakers:
            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), (metric,), 
                ax=ax, label=speaker
            )

        ax.set_title(f"Learning curve {expdir.name} {metric}")
        plt.legend()
        plt.savefig(savedir / f"curve_{metric}.{fmt}", dpi=dpi)

    if display:
        plt.show()

    plt.close("all")


@cli.command()
@click.argument("expdirs", nargs=-1)
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.option("--metric", default="f1", type=str, help="Metric to plot")
@click.option("--remove_incomplete", is_flag=True, help="Remove experiments not performed for all speakers")
@click.option("--savedir", type=click.Path(), help="Output directory")
@click.pass_context
def compare_results(ctx, expdirs, plot_speakers, metric, remove_incomplete, savedir):
    fmt, dpi, display = ctx.obj["FORMAT"], ctx.obj["DPI"], ctx.obj["DISPLAY"]
    expdirs = list(map(validate_dir, expdirs))
    exp_names = list(map(lambda p: p.name, expdirs))
    savedir = Path(savedir)
    if savedir.exists() and not ctx.obj["OVERWRITE"]:
        raise ValueError(f"{savedir} exists already. Add --overwrite to overwrite.")
    os.makedirs(savedir, exist_ok=True)

    metrics = (metric,)

    results = pd.concat([
        load_results(expdir, metrics).rename(columns={metric: exp_name}) 
        for exp_name, expdir in zip(exp_names, expdirs)
    ], axis=1)

    results = results.loc[:, ~results.columns.duplicated()]
    speakers = sorted(results.index.levels[0])

    if remove_incomplete:
        results = results.dropna(how="any")

    if plot_speakers:
        for speaker in speakers:

            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), exp_names
            )

            ax.set_title(f"Compare experiences {metric} {speaker}")
            plt.legend()
            plt.savefig(savedir / f"compare_{speaker}.{fmt}", dpi=dpi)

    ax = None
    for exp_name in exp_names:
        ax = plot_learning_curve(results, (exp_name,), label=exp_name, ax=ax)

    ax.set_title(f"Compare experiences {metric}")
    plt.legend()
    plt.savefig(savedir / f"compare_{metric}.{fmt}", dpi=dpi)

    if display:
        plt.show()

    plt.close("all")


def plot_learning_curve(dataframe, metrics, ax=None, label=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))

    _smooth = lambda y, x: lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=1/3,
        it=0,
        delta=1.,
        return_sorted=True
    )

    x = dataframe["train_size"]
    for metric in metrics:
        y = dataframe[metric]
        x, y = map(np.array, zip(*_smooth(y, x)))
        ax.plot(x, y, label=label or metric)

    return ax


def validate_dir(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)
    return directory


def read_file(filename):
    with open(filename) as f:
        return list(map(str.strip, f.readlines()))


def line_count(filename):
    with open(filename) as f:
        return len(list(filter(bool, map(str.strip, f.readlines()))))


def load_results(expdir, metrics):

    blockdirs = sorted(expdir.rglob("*blocks_exp*"))
    scores = [
        {metric: float(read_file(blockdir / metric)[0]) for metric in metrics}
        for blockdir in blockdirs
    ]
    rgx = re.compile(r"^(?P<nblocks>\d+)blocks_exp(?P<expid>\d+)$")
    results = pd.DataFrame.from_dict(scores, orient="columns")
    
    speakers = pd.Series(
        list(map(lambda d: d.parent.name, blockdirs)), 
        name="speaker"
    )

    blocks = (
        pd.Series(list(map(lambda f: f.name, blockdirs)))
        .str.extract(rgx).astype(int)
    )

    results["train_size"] = list(map(
        line_count, map(
            lambda p: p / "traintasks", blockdirs
        )
    ))

    results = pd.concat([speakers, results, blocks], axis=1)
    results.set_index(["speaker", "nblocks", "expid"], inplace=True)
    return results.sort_index()


if __name__ == "__main__":
    cli(obj={})