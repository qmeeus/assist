import argparse
import h5py
import json
import numpy as np
import os
import sys
import torch

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from time import time
from torch.utils.data import DataLoader

from assist.acquisition.torch_models import (
    SequenceDataset, 
    AttentiveDecoder, 
    AttentiveRecurrentDecoder, 
    RNN,
    display_model
)

from assist.tasks import Structure, read_task
from assist.tools import read_config
from assist.tasks import coder_factory


AVAILABLE_MODELS = {
    "att": AttentiveDecoder,
    "att_rnn": AttentiveRecurrentDecoder,
    "rnn": RNN
}


def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


def compute_error_rate(labels, logits):
    assert labels.shape == logits.shape
    predictions = np.rint(sigmoid(logits))
    return 1 - (labels == predictions).all(-1).mean()


def train(model, train_set, valid_set, checkpoint, max_epochs=20, batch_size=64, es_patience=5, es_criterion="val_loss", es_lower_is_better=True, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    best_error_rate = float("inf")
    best_loss = float("inf")
    best_epoch = 0
    epoch = 0
    t0 = time()

    def should_stop(epoch):
        return (epoch - best_epoch) > es_patience or epoch >= max_epochs

    def check_early_stopping(epoch, loss, error_rate):
        nonlocal best_error_rate, best_loss, best_epoch
        if es_criterion == "val_loss":
            metric, prev = loss, best_loss
        else:
            metric, prev = error_rate, best_error_rate

        if (metric - prev) * (-1 if es_lower_is_better else 1) > 0:
            best_epoch = epoch
            best_loss = loss
            best_error_rate = error_rate
            torch.save(model, checkpoint)


    while True:
        train_lr, val_lr = (DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not i,
            collate_fn=dataset.data_collator
        ) for i, dataset in enumerate([train_set, valid_set]))

        train_loss, val_loss = 0, 0
        model.train()
        predictions, target = None, None
        for *inputs, labels in train_lr:
            optimizer.zero_grad()
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)
            loss, preds = model(*inputs, labels=labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            predictions = (
                preds if predictions is None
                else np.concatenate([predictions, preds], axis=0)
            )
            labels = labels.detach().cpu().numpy()
            target = (
                labels if target is None
                else np.concatenate([target, labels], axis=0)
            )

        train_loss = float(train_loss/len(train_set))
        train_error_rate = compute_error_rate(target, predictions)

        predictions, target = None, None
        with torch.no_grad():
            model.eval()
            for *inputs, labels in val_lr:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                loss, preds = model(*inputs, labels=labels)
                val_loss += loss.item()
                preds = preds.detach().cpu().numpy()
                predictions = (
                    preds if predictions is None
                    else np.concatenate([predictions, preds], axis=0)
                )
                labels = labels.detach().cpu().numpy()
                target = (
                    labels if target is None
                    else np.concatenate([target, labels], axis=0)
                )

        val_loss = float(val_loss/len(valid_set))
        val_error_rate = compute_error_rate(target, predictions)

        epoch += 1
        print(f"Epoch {epoch} [ TRAIN {train_loss:.4f} {train_error_rate:.4f} ] "
              f"[ VALID {val_loss:.4f} {val_error_rate:.4f} ]")

        check_early_stopping(epoch, val_loss, val_error_rate)

        if should_stop(epoch):
            break

    print(f"Training completed in {time() - t0:.2f} s. "
          f"Best epoch: {best_epoch} Loss: {best_loss:.3f} ER: {best_error_rate:.4%}")


# def load_fluent(options):

#     taskfiles = list(Path("/users/spraak/spchdata/FluentSpeechCommands/assist").rglob("tasks"))
    
#     def load_tasks(filename):
#         with open(filename) as f:
             
#             uttids, tasks = map(list, zip(*map(
#                 lambda s: s.split(maxsplit=1), 
#                 map(str.strip, f.readlines())
#             )))
#         spkr = filename.parent.name
#         return list(map(f"{spkr}_{{}}".format, uttids)), tasks

#     uttids, tasks = [], []
#     for filename in taskfiles:        
#         us, ts = load_tasks(filename)
#         uttids.extend(us)
#         tasks.extend(ts)

#     coderconf = read_config(options.expdir/"coder.cfg")
#     structure = Structure(options.expdir/'structure.xml')
#     Coder = coder_factory(coderconf.get("coder", "name"))
#     coder = Coder(structure, coderconf)
#     encoded_tasks = np.array(list(map(coder.encode, map(read_task, tasks))))
#     featconf = read_config(options.expdir/"features.cfg")
#     featfile = Path(featconf.get("features", "file"))
#     with h5py.File(featfile, "r") as f:
#         features = [f[uttid][()] for uttid in uttids]

#     features = np.array(features, dtype="object")
#     uttids = np.array(uttids)
#     train_mask = np.array(list(map(lambda s: "_train_" in s, uttids)))
#     train_set = SequenceDataset(features[train_mask], encoded_tasks[train_mask])

#     valid_mask = np.array(list(map(lambda s: "_valid_" in s, uttids)))
#     valid_set = SequenceDataset(features[valid_mask], encoded_tasks[valid_mask])

#     # test_mask = np.array(list(map(lambda s: "_test_" in s, uttids)))
#     # test_set = SequenceDataset(features[test_mask], encoded_tasks[test_mask])

#     return train_set, valid_set


def load_data(options):

    structure = Structure(options.expdir/"structure.xml")
    coderconf = read_config(options.expdir/"coder.cfg")
    Coder = coder_factory(coderconf.get("coder", "name"))
    coder = Coder(structure, coderconf)
    dataconf = read_config(options.expdir/"database.cfg")

    def load_tasks(filename):
        with open(filename) as f:
            spkr = filename.stem
            
            if spkr == "tasks":
                spkr = filename.parent.name
            
            uttids, tasks = map(list, zip(*map(
                lambda s: s.split(maxsplit=1), 
                map(str.strip, f.readlines()))
            ))

        return list(map(f"{spkr}_{{}}".format, uttids)), tasks

    features = {
        uttid: feats
        for spkr in dataconf.sections()
        for uttid, feats in np.load(dataconf[spkr].get("features")).items()
    }

    labels = {
        uttid: coder.encode(read_task(task))
        for spkr in dataconf.sections()
        for uttid, task in zip(*load_tasks(Path(dataconf[spkr].get("tasks"))))
    }

    errors = set(features).union(set(labels)) - set(features).intersection(set(labels))
    if errors:
        msg = f"{len(errors)} mismatches ({len(features)} features and {len(labels)} labels)"
        if options.errors == "raise":
            raise Exception(msg)
        elif options.errors == "warn":
            warning.warn(msg)

    uttids = np.array(list(features))
    features = np.array([features[uttid] for uttid in uttids], dtype="object")
    labels = np.array([labels[uttid] for uttid in uttids])

    if any(subset in uttids[0] for subset in ["train", "valid", "test"]):
        train_mask = np.array(list(map(lambda s: "_train_" in s, uttids)))
        valid_mask = np.array(list(map(lambda s: "_valid_" in s, uttids)))
        # test_mask = np.array(list(map(lambda s: "_test_" in s, uttids)))

    elif (options.expdir/"train.cfg").exists():
        train_sections = set(read_config(options.expdir/"train.cfg").get("train", "datasections").split())
        test_sections = set(read_config(options.expdir/"test.cfg").get("test", "datasections").split())
        train_mask = np.array(list(map(lambda s: s.split("_")[0] in train_sections, uttids)))
        valid_mask = np.array(list(map(lambda s: s.split("_")[0] in test_sections, uttids)))

    else:
        train_ids, valid_ids = train_test_split(uttids)
        train_mask = np.array(list(map(lambda s: s in train_ids, uttids)))
        valid_mask = np.array(list(map(lambda s: s in valid_ids, uttids)))

    train_set = SequenceDataset(features[train_mask], labels[train_mask])
    valid_set = SequenceDataset(features[valid_mask], labels[valid_mask])
    # test_set = SequenceDataset(features[test_mask], labels[test_mask])

    return train_set, valid_set 



def build_model(input_dim, output_dim, options):
    with open(options.model_config or options.expdir/"model.json") as f:
        config = json.load(f)
    
    model_name = config.pop("name")
    if model_name not in AVAILABLE_MODELS:
        raise TypeError(f"{model_name} is not one of {list(AVAILABLE_MODELS)}")
    Model = AVAILABLE_MODELS[model_name]

    model = Model(train_set.input_dim, train_set.output_dim, **config)
    display_model(model)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path, help="exp/fluent/train_lstm_128")
    parser.add_argument("expdir", type=Path, help="config/FluentSpeechCommands/lstm_128")
    parser.add_argument("--model-config", type=Path, help="JSON with model specific config")
    parser.add_argument("--config", type=Path, nargs="+", help="JSON with train specific config")
    parser.add_argument("--errors", choices=["ignore", "raise", "warn"], default="raise")
    parser.add_argument("--device", type=torch.device, default="cuda")
    options = parser.parse_args()

    checkpoint = options.outdir/"model.pt"
    os.makedirs(options.outdir, exist_ok=True)
    device = torch.device(options.device)
    train_set, valid_set = load_data(options)

    model = build_model(train_set.input_dim, train_set.output_dim, options)

    config = {}
    for cfg in options.config:
        with open(cfg) as f:
            config.update(json.load(f))

    train(model, train_set, valid_set, checkpoint, device=device, **config)
