import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append("/users/spraak/qmeeus/.local/share/src/dotfiles/scripts/python/data-science")
from tsne_nlp_feats import load_features, load_data, plot_tsne


targetfile = Path("/users/spraak/qmeeus/spchtemp/repos/espnet/egs/grabo/sti1/data/grabo_w2v/encoded_target.csv")
featfile = Path("/users/spraak/qmeeus/spchtemp/repos/espnet-stable/egs/cgn/asr1/exp/CGN_train_pytorch_sti_transformer_lr10.0_ag8_p.5_mlm_specaug/encode/grabo/predictions.h5")
TARGET = pd.read_csv(targetfile, delimiter=",", index_col="uttid")
TARGET.columns = list(map(lambda col: col.replace("pos_", "pos"), TARGET.columns))
TYPES = defaultdict(list)
for typ in TARGET.columns:
    key, value = typ.split("_", maxsplit=1)
    TYPES[key].append(typ)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--featfile", type=Path, default=featfile)
    parser.add_argument("--perplexity", type=int, default=100)
    parser.add_argument(
        "--action", default="all", type=str,
        choices=["all", *map(lambda s: s.replace("action_", ""), TYPES["action"])])
    parser.add_argument("--save-as", default=None, type=Path)
    return parser.parse_args()


def main():
    options = parse_args()

    features = load_features(TARGET, options.featfile)

    if options.action == "all":
        mask = [True] * len(TARGET)
        classes = TYPES["action"]
    else:
        mask = TARGET[f"action_{options.action}"] == 1
        classes = TARGET[mask].sum(0).where(lambda x: x>0).dropna().index[1:].tolist()

    feats = features[mask]
    target = TARGET.loc[mask, classes].apply(
        lambda row: "_".join(map(lambda s: s.split("_", maxsplit=1)[1], row[row == 1].index)), axis=1
    )

    encoder = LabelEncoder()
    tgt = encoder.fit_transform(target)
    cmap = "tab20" if len(encoder.classes_) > 8 else "muted"

    plot_tsne(feats, tgt, target_names=encoder.classes_, perplexity=options.perplexity, cmap=cmap, random_state=42)
    if options.save_as:
        os.makedirs(options.save_as.parent, exist_ok=True)
        plt.savefig(options.save_as)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
