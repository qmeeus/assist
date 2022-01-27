# coding: utf-8
from pathlib import Path


def prepare_for_evaluation(model_dir):
    dirs = list(model_dir.glob("split*"))
    with open(dirs[0].parent/"tasks") as f:
        tasks = f.readlines()
    with open(dirs[0].parent/"feats") as f:
        feats = f.readlines()
    for d in dirs:
        for subset in ("train", "test"):
            with open(d/f"{subset}indices") as f:
                indices = list(map(str.strip, f.readlines()))
            with open(d/f"{subset}feats", "w") as f:
                f.writelines(filter(lambda s: s.split(maxsplit=1)[0] in indices, feats))
            with open(d/f"{subset}tasks", "w") as f:
                f.writelines(filter(lambda s: s.split(maxsplit=1)[0] in indices, tasks))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    options = parser.parse_args()
    prepare_for_evaluation(**vars(options))
