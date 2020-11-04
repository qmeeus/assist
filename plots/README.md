# Plotting Interface

**NB: python3 only (python2 is dead anyway)**

```bash
$ python -m plots

Usage: __main__.py [OPTIONS] COMMAND [ARGS]...

Options:
  --format [png|eps]
  --dpi INTEGER
  --overwrite
  --display
  --help              Show this message and exit.

Commands:
  compare-results
  learning-curve
```

```bash
$ python -m plots learning-curve --help
Usage: __main__.py learning-curve [OPTIONS] [METRICS]...

Options:
  --expdir PATH    [required]
  --plot_speakers  Plot speakers separately
  --help           Show this message and exit.
```

```bash
python -m plots compare-results --help
Usage: __main__.py compare-results [OPTIONS] [EXPDIRS]...

Options:
  --plot_speakers      Plot speakers separately
  --metric TEXT        Metric to plot
  --remove_incomplete  Remove experiments not performed for all speakers
  --savedir PATH       Output directory
  --help               Show this message and exit.
```
