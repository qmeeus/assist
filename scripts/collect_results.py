# coding: utf-8
import numpy as np
from pathlib import Path


def get_metric(fn):
    with open(fn) as f:
        return float(f.read())


def collect_results(model_dir, metric="accuracy"):
    dirs = options.model_dir.glob("split*")
    metrics = [get_metric(d/options.metric) for d in dirs]
    score_mean = np.mean(metrics)
    score_std = np.std(metrics)
    print(f"{options.metric}: {score_mean:.4f} ({score_std:.4f})")
    return metrics, (score_mean, score_std)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("-m", "--metric", default="accuracy")
    options = parser.parse_args()
    collect_results(**vars(options))

