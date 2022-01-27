import json
import kaldiio
import numpy as np
import os
from itertools import groupby
from pathlib import Path
from assist.tools.io import save_features

ROOT = Path("/esat/spchtemp/scratch/qmeeus/repos")
SRC_FILE = ROOT/"espnet-stable/egs/cgn/asr1/dump/grabo_patience/nopitch/data.grabo.json"
DEST_DIR = ROOT/"assist/features/grabo/fbank"

with open(SRC_FILE) as f:
    grabo_data = json.load(f)["utts"]

uttids = list(grabo_data)

os.makedirs(DEST_DIR, exist_ok=True)
for speaker, group in groupby(uttids, lambda uttid: uttid.split("_")[0]):
    arrays = {
        uttid: kaldiio.load_mat(grabo_data[uttid]["input"][0]["feat"])
        for uttid in group
    }
    save_features(arrays, DEST_DIR/f"{speaker}.npz", storage="numpy", feature_name="fbank")

