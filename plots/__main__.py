import click
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp

from functools import partial
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess


@click.group()
@click.option("--format", default="png", type=click.Choice(["png", "eps"]))
@click.option("--dpi", default=300, type=int)
@click.option("--display", is_flag=True)
@click.option("--savedir", type=click.Path(), default="exp/figures")
@click.pass_context
def cli(ctx, format, dpi, display, savedir):
    ctx.ensure_object(dict)
    ctx.obj["FORMAT"] = format
    ctx.obj["DPI"] = dpi
    ctx.obj["DISPLAY"] = display
    ctx.obj["SAVEDIR"] = Path(savedir)
    os.makedirs(savedir, exist_ok=True)


@cli.command()
@click.argument("expdir", type=click.Path(exists=True))
@click.option("-m", "--metrics", multiple=True, default=["f1"])
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.option("--save_as", type=str, help="Output file", default="learning_curve_{dataset}_{model}")
@click.option("--overwrite", is_flag=True)
@click.pass_context
def learning_curve(ctx, expdir, metrics, plot_speakers, save_as, overwrite):
    
    fmt = ctx.obj["FORMAT"]
    dpi = ctx.obj["DPI"]
    display = ctx.obj["DISPLAY"]
    savedir = ctx.obj["SAVEDIR"]
    expdir = Path(expdir)

    results = load_results(expdir, metrics)
    speakers = sorted(results.index.levels[0])
    save_as = save_as.format(dataset=expdir.parent.name, model=expdir.name)

    if plot_speakers:
        
        for speaker in speakers:

            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), metrics
            )

            ax.set_title(f"Learning curve {expdir.name} {speaker}")
            plt.legend()
            outfile = savedir/f"{save_as}_{speaker}.{fmt}"
            savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    for metric in metrics:
        ax = None

        for speaker in speakers:
            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), (metric,),
                ax=ax, label=speaker
            )

        ax.set_title(f"Learning curve {expdir.name} {metric}")
        plt.legend()
        outfile = savedir/f"{save_as}_all_{metric}.{fmt}"
        savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    display and plt.show()
    plt.close("all")



@cli.command()
@click.argument("expdirs", nargs=-1, type=click.Path(exists=True))
@click.option("-m", "--metrics", multiple=True, default=["f1"])
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.option("--remove_incomplete", is_flag=True, help="Remove experiments not performed for all speakers")
@click.option("--save_as", type=str, help="Output file", default="compare_{dataset}")
@click.option("--overwrite", is_flag=True)
@click.pass_context
def compare_results(ctx, expdirs, metrics, plot_speakers, remove_incomplete, save_as, overwrite):
    
    fmt = ctx.obj["FORMAT"]
    dpi = ctx.obj["DPI"]
    display = ctx.obj["DISPLAY"]
    savedir = ctx.obj["SAVEDIR"]
    expdirs = list(map(Path, expdirs))
    expnames = list(map(lambda p: p.name, expdirs))

    save_as = save_as.format(dataset=expdirs[0].parent.name)

    results = pd.concat([load_results(expdir, metrics) for expdir in expdirs], axis=1)
    results.columns = pd.MultiIndex.from_product([expnames, metrics + ("train_size",)])
    results = results.loc[:, ~results.columns.duplicated()]
    speakers = sorted(results.index.levels[0])

    if remove_incomplete:
        results = results.dropna(how="any")

    if plot_speakers:
        for speaker in speakers:

            view_spkr = results.xs(speaker, level="speaker", axis=0)

            ax = None
            for expname in expnames:
                view_exp = view_spkr.xs(expname, level=0, axis=1)
                ax = plot_learning_curve(view_exp, metrics, ax=ax)

            ax.set_title(f"Compare {len(expnames)} experiments on {metrics} for {speaker}")
            plt.legend()
            outfile = savedir/f"{save_as}_{speaker}.{fmt}"
            savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    ax = None
    for expname in expnames:
        view_exp = results.xs(expname, level=0, axis=1)
        labels = [f"{expname} ({metric})" for metric in metrics]
        ax = plot_learning_curve(view_exp, metrics, labels=labels, ax=ax)

    ax.set_title(f"Compare {len(expnames)} experiments on {metrics} for all speakers")
    plt.legend()
    savefig(plt.gcf(), savedir / f"{save_as}_all.{fmt}", overwrite=overwrite, dpi=dpi)
    display and plt.show() or plt.close("all")


def plot_learning_curve(dataframe, metrics, labels=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))

    _smooth = lambda y, x: lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=1/3,
        it=0,
        delta=1.,
        return_sorted=True
    )

    if type(labels) is not list:
        labels = [labels] * len(metrics)

    x = dataframe["train_size"]
    for metric, label in zip(metrics, labels):
        y = dataframe[metric]
        x, y = map(np.array, zip(*_smooth(y, x)))
        ax.plot(x, y, label=label or metric)

    return ax


def savefig(fig, filename, overwrite=False, dpi=300):
    fig.tight_layout()
    if Path(filename).exists() and not overwrite:
        raise ValueError(f"{filename} exists and overwrite flag not set.")
    fig.savefig(filename, dpi=dpi)


def read_file(filename):
    with open(filename) as f:
        lines = list(map(str.strip, f.readlines()))
        return lines[0] if len(lines) == 1 else lines


def line_count(filename):
    with open(filename) as f:
        return len(list(filter(bool, map(str.strip, f.readlines()))))


def load_results(expdir, metrics):

    blockdirs = sorted(expdir.rglob("*blocks_exp*"))
    with mp.Pool(mp.cpu_count()) as pool:
        scores = pd.DataFrame({
            metric: pool.map(read_file, [blockdir/metric for blockdir in blockdirs])
            for metric in metrics
        })

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
