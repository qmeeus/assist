import h5py
import numpy as np
import os

from functools import partial
from operator import itemgetter
from pathlib import Path

from assist.tools import (
    load_hdf5,
    logger,
    parse_line,
    read_config,
    save_features,

)


__all__ = [
    "prepare_features",
    "map_prepare_features",
]


def prepare_features(expdir):

    expdir = Path(expdir)
    featureconf_file = expdir/"features.cfg"
    dataconf_file = expdir/"database.cfg"

    dataconf = dict(read_config(dataconf_file).items("database"))
    featconf = read_config(featureconf_file)
    os.makedirs(Path(dataconf['features']).parent, exist_ok=True)

    aggfunc = {
        "average": partial(np.mean, axis=0),
        "first": itemgetter(0)
    }.get(featconf.get("features", "agg"), None)

    convert = {
        "precomputed": convert_hdf5,
        "text": encode_feats
    }.get(featconf.get("features", "name"), compute_feats)
    logger.info(f"Feature created with {convert.__name__}")
    convert(expdir, featconf, dataconf, aggfunc=aggfunc)


def map_prepare_features(args):
    return prepare_features(*args)


def convert_hdf5(expdir, featconf, dataconf, aggfunc=None):

    speaker = expdir.name
    source_file = Path(featconf.get("features", "file"))
    target_file = Path(dataconf["features"])
    storage_type = featconf.get("features", "storage")

    data = load_hdf5(source_file, filter_keys=lambda s: s.startswith(speaker))
    if aggfunc is not None:
        data = {key: aggfunc(value) for key, value in data.items()}

    save_features(data, target_file, storage_type)


def encode_feats(expdir, featconf, dataconf, aggfunc=None):

    speaker = expdir.name
    target_file = Path(dataconf["features"])
    storage_type = featconf.get("features", "storage")
    model_string = featconf.get("features", "encoder")

    import torch
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Loading tokenizer and model: {model_string}")
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    encoder = AutoModel.from_pretrained(model_string)
    encoder.eval()

    # HACK: Small workaround but would be better to use --cuda flag
    device = torch.device(
        "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", None) else "cpu"
    )

    logger.debug(f"Device: {device}")
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    with open(featconf.get("features", "file"), "r") as textfile:
        uttids, texts = map(list, zip(
            *map(parse_line, filter(
                lambda line: line.startswith(speaker),
                textfile.readlines()
            ))
        ))

    inputs = tokenizer(
        texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    )

    input_lengths = inputs["attention_mask"].sum(-1)

    out = encoder(**{
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in inputs.items()
    })

    if len(out) == 2:
        data = out[featconf.get("features", "agg") == "pooled"]
    else:
        data, = out

    data = {
        uttid: data[i, :length].cpu().numpy()
        for i, (uttid, length) in enumerate(zip(uttids, input_lengths))
    }
    
    if aggfunc is not None:
        data = {uttid: aggfunc(array) for uttid, array in data.items()}

    save_features(data, target_file, storage_type)


def compute_feats(expdir, featconf, dataconf):

    logger.error("Deprecated method")
    from assist.features import feature_computer_factory
    from assist.features.read_audio import read_audio

    feature_computer = feature_computer_factory.factory(
        featconf.get('features', 'name'))(featconf)

    with open(dataconf["audio"]) as scpfile:
        uttids, audiofiles = zip(*map(parse_line, scpfile.readlines()))

    featdir = Path(dataconf["features"])
    filenames = [featdir/f"{filename}.npy" for uttid in uttids]

    for filename, audiofile in zip(filenames, audiofiles):
        rate, sig = read_audio(wavfile)
        if sig.ndim == 2:
            # average channels
            sig = np.int16(np.mean(sig, axis=1))
        feats = feature_computer(sig, rate)
        np.save(filename, feats)

    with open(Path(Path(filenames[0]).parent, 'feats'), 'w') as featfile:
        featfile.writelines([
            f"{uttid} {filename}\n" for uttid, filename in zip(uttids, filenames)
        ])
