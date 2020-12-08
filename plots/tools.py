import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import re
import sys

from functools import partial
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess

from .logger import logger


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
    logger.info(f"Save figure as {filename}")
    fig.tight_layout()
    if Path(filename).exists() and not overwrite:
        logger.error(f"{filename} exists and overwrite flag not set. The figure will not be saved.")
        return
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
    logger.info(f"{len(results)} results loaded from {expdir}")
    return results.sort_index()
