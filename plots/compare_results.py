'''@file compare_results.py
compare the results of experiments on the same database

usage: python compare_results.py result expdir1, expdir2, ...
    expdir: the experiments directory of one of the experiments
    result: what you want to plot (e.g. f1)
'''

import sys
import os
import itertools
import numpy as np
import datetime as dt
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns
from plots.get_results import get_results


def main(expdirs, metric):
    '''main function'''

    expdirs = [os.path.normpath(expdir) for expdir in expdirs]

    # colorlist = ['red', 'blue', 'cyan', 'green', 'yellow', 'magenta',
    #             'purple', 'pink', 'gold', 'navy', 'olive', 'grey']
    #linestyles = ['-']

    palette = sns.color_palette(n_colors=len(expdirs))
    colorlist = ['black']
    linestyles = ['-', '--', ':', '-.']

    #colorlist = ['green', 'dimgrey', 'darkorange']
    #linestyles = ['-']

    plot_speakers = False
    remove_uncomplete = True

    #tick parameters
    tick_params = {
        'size': 'x-large',
        #'color': 'dimgrey'
    }

    #axis properties
    ax_params = {
        # 'color':'black'
        # "c": "k"
    }

    #label properties
    label_params = {
        'color':'black',
        'size': 'x-large'
    }

    #legend properties
    legend_params = {
        'loc': 'lower right',
        'edgecolor':'black',
        'fontsize': 'x-large'
    }
    lcolor = 'black'

    #lowess parameters
    smooth = lambda y, x: lowess(
        y, x + 1e-12 * np.random.randn(len(x)),
        frac=2.0/3,
        it=0,
        delta=1.0,
        return_sorted=True)


    #read all the results
    results = [get_results(expdir, metric) for expdir in expdirs]
    expnames = [os.path.basename(expdir) for expdir in expdirs]

    #remove experiments that are not performed in all experiments
    if remove_uncomplete:
        speakers = set(results[0].keys())
        for result in results[1:]:
            speakers = speakers & set(result.keys())
        results = [{s: result[s] for s in speakers} for result in results]
        for speaker in speakers:
            experiments = set(results[0][speaker].keys())
            for result in results[1:]:
                experiments = experiments & set(result[speaker].keys())
            if not experiments:
                for result in results:
                    del result[speaker]
            else:
                for result in results:
                    result[speaker] = {
                        e: result[speaker][e] for e in experiments}


    if plot_speakers:
        for speaker in results[0]:
            plt.figure(speaker)
            for i, result in enumerate(results):
                if speaker not in result:
                    continue
                sort = np.array(list(result[speaker].values()))
                sort = sort[np.argsort(sort[:, 0], axis=0), :]
                fit = smooth(sort[:, 1], sort[:, 0])
                plot = plt.plot(
                    fit[:, 0], fit[:, 1],
                    color=colorlist[i%len(colorlist)],
                    linestyle=linestyles[i%len(linestyles)],
                    label=expnames[i])
            plt.yticks(**tick_params)
            plt.xticks(**tick_params)
            plt.axis(**ax_params)
            l = plt.legend(**legend_params)
            for text in l.get_texts():
                text.set_color(lcolor)
            plt.xlabel('# Examples', **label_params)
            plt.ylabel('Accuracy', **label_params)

    #concatenate all the results
    concatenated = [
        np.array(list(itertools.chain.from_iterable(
            [r.values() for r in result.values()])))
        for result in results]

    #sort the concatenated data
    sort = [c[np.argsort(c[:, 0], axis=0), :] for c in concatenated]

    #smooth all the results
    fit = [smooth(s[:, 1], s[:, 0]) for s in sort]

    markers = list(".+xd^v")
    display_names = "MLM,TS,TNM,noKL,MSE".split(",")

    plt.figure('result')
    for i, f in enumerate(fit):
        plt.plot(
            f[:, 0], f[:, 1],
            #  color=colorlist[i%len(colorlist)],
            color=palette[i],
            # linestyle=linestyles[i%len(linestyles)],
            marker=markers[i%len(markers)],
            label=display_names[i]
        )

    plt.yticks(**tick_params)
    plt.xticks(**tick_params)
    plt.axis(**ax_params)
    l = plt.legend(**legend_params)
    for text in l.get_texts():
        text.set_color(lcolor)
    plt.xlabel('Num train examples', **label_params)
    plt.ylabel(metric, **label_params)
    fn = f"exp/figures/compare_results_{dt.datetime.now():%Y%m%d%H%M%S}.png"
    plt.tight_layout()
    plt.savefig(fn)
    print(f"Saved to {fn}")
    plt.show()


if __name__ == '__main__':
    main(sys.argv[2:], sys.argv[1])

