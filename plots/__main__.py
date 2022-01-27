import click
import matplotlib as mpl
import os
import pandas as pd

from pathlib import Path


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", count=True)
@click.option("--format", default="png", type=click.Choice(["png", "eps"]))
@click.option("--dpi", default=300, type=int)
@click.option("--display", is_flag=True)
@click.option("--savedir", type=click.Path(), default="exp/figures")
@click.pass_context
def cli(ctx, verbose, format, dpi, display, savedir):
    verbose = min(max(0, verbose), 3)
    os.environ["LOGLEVEL"] = {
        0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"
    }.get(verbose)

    if not display:
        mpl.use("Agg")

    ctx.ensure_object(dict)
    ctx.obj["FORMAT"] = format
    ctx.obj["DPI"] = dpi
    ctx.obj["DISPLAY"] = display
    ctx.obj["SAVEDIR"] = Path(savedir)
    os.makedirs(savedir, exist_ok=True)
    click.echo("LOGLEVEL: " + os.environ["LOGLEVEL"])


@cli.command()
@click.argument("expdir", type=click.Path(exists=True))
@click.option("-m", "--metrics", multiple=True, default=["f1"])
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.option("--saveas", type=str, help="Output file", default="learning_curve_{dataset}_{model}")
@click.option("--overwrite", is_flag=True)
@click.pass_context
def learning_curve(ctx, expdir, metrics, plot_speakers, saveas, overwrite):

    import matplotlib.pyplot as plt
    from .tools import load_results, plot_learning_curve, savefig

    fmt = ctx.obj["FORMAT"]
    dpi = ctx.obj["DPI"]
    display = ctx.obj["DISPLAY"]
    savedir = ctx.obj["SAVEDIR"]
    expdir = Path(expdir)

    results = load_results(expdir, metrics)
    speakers = sorted(results.index.levels[0])
    saveas = saveas.format(dataset=expdir.parent.name, model=expdir.name)

    if plot_speakers:

        for speaker in speakers:

            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), metrics
            )

            ax.set_title(f"Learning curve {expdir.name} {speaker}")
            plt.legend()
            outfile = savedir/f"{saveas}_{speaker}.{fmt}"
            savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    for metric in metrics:
        ax = None

        for speaker in speakers:
            ax = plot_learning_curve(
                results.xs(speaker, level="speaker"), (metric,),
                ax=ax, labels=speaker
            )

        ax.set_title(f"Learning curve {expdir.name} {metric}")
        plt.legend()
        outfile = savedir/f"{saveas}_all_{metric}.{fmt}"
        savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    display and plt.show()
    plt.close("all")



@cli.command()
@click.argument("expdirs", nargs=-1, type=click.Path(exists=True))
@click.option("-m", "--metrics", multiple=True, default=["f1"])
@click.option("--plot_speakers", is_flag=True, help="Plot speakers separately")
@click.option("--remove_incomplete", is_flag=True, help="Remove experiments not performed for all speakers")
@click.option("--saveas", type=str, help="Output file", default="compare_{dataset}")
@click.option("--overwrite", is_flag=True)
@click.option("--display-names", type=str, default=None, help="List of names for legend, separated with ,")
@click.pass_context
def compare_results(ctx, expdirs, metrics, plot_speakers, remove_incomplete, saveas, overwrite, display_names):

    import matplotlib.pyplot as plt
    from .tools import load_results, plot_learning_curve, savefig

    fmt = ctx.obj["FORMAT"]
    dpi = ctx.obj["DPI"]
    display = ctx.obj["DISPLAY"]
    savedir = ctx.obj["SAVEDIR"]
    expdirs = list(map(Path, expdirs))
    expnames = list(map(lambda p: p.name, expdirs))

    if display_names:
        display_names = display_names.split(",")
        if len(expdirs) != len(display_names):
            raise click.BadParameter("Wrong number of names provided", param=display_names)

    saveas = saveas.format(dataset=expdirs[0].parent.name)

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
            outfile = savedir/f"{saveas}_{speaker}.{fmt}"
            savefig(plt.gcf(), outfile, overwrite=overwrite, dpi=dpi)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    markers = list(".+xd^v")
    linestyles = ["-", "--", ":", "-."]
    assert len(markers) >= len(expnames)
    labels = None
    for i, expname in enumerate(expnames):
        view_exp = results.xs(expname, level=0, axis=1)
        labels = [f"{expname} ({metric})" for metric in metrics] if not display_names else display_names[i]
        ax = plot_learning_curve(view_exp, metrics, linestyles=linestyles[i%len(linestyles)], labels=labels, ax=ax)

    # ax.set_title(f"Compare f1-score at increasing training sizes \n{len(expnames)} speaker-dependent experiments")
    plt.legend()
    savefig(fig, savedir / f"{saveas}_all.{fmt}", overwrite=overwrite, dpi=dpi)
    display and plt.show()
    plt.close("all")


if __name__ == "__main__":
    cli(obj={})
