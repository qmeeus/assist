import argparse
import h5py
import json
import kaldiio
import numpy as np
import os
import shutil
import sys
import torch

from copy import deepcopy
from functools import partial
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from time import time
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset

import logzero
from logzero import logger

from assist.acquisition.torch_models import (
    SequenceDataset,
    AttentiveDecoder,
    AttentiveRecurrentDecoder,
    RNN,
    display_model
)

from assist.tasks import Structure, read_task
from assist.tools import logger, read_config, FeatLoader, parse_line
from assist.tasks import coder_factory


AVAILABLE_MODELS = {
    "att": AttentiveDecoder,
    "att_rnn": AttentiveRecurrentDecoder,
    "rnn": RNN
}

DEFAULT_OPTIMIZER = ("Adam", {"lr": 1e-2})
DEFAULT_SCHEDULER = (None, {})

def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


def compute_error_rate(labels, logits):
    assert labels.shape == logits.shape
    predictions = np.rint(sigmoid(logits))
    return 1 - (labels == predictions).all(-1).mean()


def get_optimizer(name, config):
    Optimizer = getattr(torch.optim, name)
    return partial(Optimizer, **config)

def get_scheduler(optimizer, name, config):
    if not name:
        return
    Scheduler = getattr(lr_scheduler, name)
    return Scheduler(optimizer, **config)


def train(model,
          train_set,
          valid_set,
          checkpoint,
          max_epochs=20,
          batch_size=64,
          es_patience=5,
          es_criterion="val_loss",
          es_lower_is_better=True,
          optimizer=DEFAULT_OPTIMIZER,
          scheduler=DEFAULT_SCHEDULER,
          device="cuda"):

    model.to(device)
    optimizer = get_optimizer(*optimizer)(model.parameters())
    scheduler = get_scheduler(optimizer, *scheduler)

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

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()

        epoch += 1
        lr = scheduler.get_lr() if hasattr(scheduler, "get_lr") else "-"
        logger.info(f"Epoch {epoch} lr={lr} "
              f"[ TRAIN {train_loss:.4f} {train_error_rate:.4f} ] "
              f"[ VALID {val_loss:.4f} {val_error_rate:.4f} ]")

        check_early_stopping(epoch, val_loss, val_error_rate)

        if should_stop(epoch):
            break

    logger.info(f"Training completed in {time() - t0:.2f} s. "
          f"Best epoch: {best_epoch} Loss: {best_loss:.3f} ER: {best_error_rate:.4%}")
    logger.info(f"Checkpoint: {checkpoint}")

    return {"error_rate": best_error_rate, "loss": best_loss, "epochs": best_epoch}


class TrainTestLoader:

    def __init__(self, options):

        self.outdir = options.outdir
        self.expdir = options.expdir
        self.method = options.method
        self.dataconf = self.load_dataconf(self.expdir)
        self.structure, self.coder = self.load_structure_and_coder(self.expdir)

        if options.train and options.valid:
            self.splits = {"train": options.train, "valid": options.valid}
            assert "train" in self.splits and "valid" in self.splits

        if self.expdir is None:
            self.expdir = options.expdir = options.outdir

        if self.splits:
            load_func = self.load_splits
        elif (self.expdir/"feats.scp").exists():
            load_func = self.load_from_scp
        elif self.expdir == self.outdir:
            load_func = self.load_from_outdir
        else:
            load_func = self.load_from_conf

        self.train_set, self.valid_set = load_func(options)

    def write_to_file(self):
        with open(self.outdir/"trainfeats") as featfile, \
                open(self.outdir/"traintasks") as taskfile:
            for uttid in self.uttids[self.train_mask]:
                speaker = "_".join(uttid.split("_")[:-1])
                featpath = self.dataconf.get("speaker")["feats"]
                taskstring = self.coder.decode(self.labels[uttid])
                featfile.write(f"{uttid} {featpath}:{uttid}\n")
                taskfile.write(f"{uttid} {taskstring}")

    @staticmethod
    def add_arguments(parser):
        parser.add_argument_group("data")
        parser.add_argument("--outdir", type=Path, help="exp/fluent/train_lstm_128")
        parser.add_argument("--expdir", type=Path, help="config/FluentSpeechCommands/lstm_128")
        parser.add_argument("--method", choices=["10-fold", "5-fold", "3-fold", "full", "10%", "1%"], default="full")
        parser.add_argument("--errors", choices=["ignore", "raise", "warn"], default="raise")
        parser.add_argument("--train", type=Path, help="Predefined splits as scp files")
        parser.add_argument("--valid", type=Path, help="Predefined splits as scp files")
        return parser

    @staticmethod
    def load_dataconf(confdir):
        dataconf = read_config(confdir/"database.cfg")
        return dataconf

    @staticmethod
    def load_structure_and_coder(expdir):
        structure = Structure(expdir/"structure.xml")
        coderconf = read_config(expdir/"coder.cfg")
        Coder = coder_factory(coderconf.get("coder", "name"))
        coder = Coder(structure, coderconf)
        return structure, coder

    def load_splits(self, options):

        options.method = "nosplit"
        train_set = self.load_from_scp(options, filename=self.splits["train"])
        valid_set = self.load_from_scp(options, filename=self.splits["valid"])
        return train_set, valid_set


    def load_from_outdir(self, options):
        outdir = self.outdir
        trainfeats = FeatLoader(outdir/"trainfeats").to_dict()
        testfeats = FeatLoader(outdir/"testfeats").to_dict()
        with open(outdir/"traintasks") as traintasks:
            trainlabels = {
                uttid: self.coder.encode(read_task(task))
                for uttid, task in map(parse_line, traintasks.readlines())
            }

        with open(outdir/"testtasks") as testtasks:
            testlabels = {
                uttid: self.coder.encode(read_task(task))
                for uttid, task in map(parse_line, testtasks.readlines())
            }

        features = set(trainfeats).union(set(testfeats))
        labels = set(trainlabels).union(set(testlabels))

        features, labels = self.check_errors(features, labels, options.errors)

        trainuttids = set(trainfeats)
        trainfeats = np.array([trainfeats[uttid] for uttid in trainuttids], dtype="object")
        trainlabels = np.array([trainlabels[uttid] for uttid in trainuttids])
        train_set = SequenceDataset(trainfeats, trainlabels)

        testuttids = set(testfeats)
        testfeats = np.array([testfeats[uttid] for uttid in testuttids], dtype="object")
        testlabels = np.array([testlabels[uttid] for uttid in testuttids])
        valid_set = SequenceDataset(testfeats, testlabels)

        return train_set, valid_set

    def load_from_conf(self, options):
        confdir = self.expdir

        features = {
            uttid: feats
            for spkr in self.dataconf.sections()
            for uttid, feats in np.load(self.dataconf[spkr].get("features")).items()
        }

        labels = {
            uttid: self.coder.encode(read_task(task))
            for spkr in self.dataconf.sections()
            for uttid, task in zip(*self.load_tasks(Path(self.dataconf[spkr].get("tasks"))))
        }

        features, labels = self.check_errors(features, labels, options.errors)
        return self.make_splits(features, labels, options)

    def load_from_scp(self, options, filename=None):

        scpfile = filename or self.expdir/"feats.scp"

        features = dict(kaldiio.load_scp(str(scpfile)))
        labels = {
            uttid: self.coder.encode(read_task(task))
            for spkr in self.dataconf.sections()
            for uttid, task in zip(*self.load_tasks(Path(self.dataconf[spkr].get("tasks"))))
        }

        features, labels = self.check_errors(features, labels, options.errors)
        return self.make_splits(features, labels, options)

    def make_splits(self, features, labels, options):
        uttids = np.array(list(features))
        features = np.array([features[uttid] for uttid in uttids], dtype="object")
        labels = np.array([labels[uttid] for uttid in uttids])

        if options.method == "nosplit":
            return SequenceDataset(features, labels)

        if options.method == "10-fold":
            return SequenceDataset(features, labels, indices=uttids), None

        # Fluent Speech Commands
        if any(subset in uttids[0] for subset in ["train", "valid", "test"]):
            logger.info("Fluent Speech Commands dataset splits")
            train_mask = np.array(list(map(lambda s: "_train_" in s, uttids)))
            valid_mask = np.array(list(map(lambda s: "_valid_" in s, uttids)))
            # test_mask = np.array(list(map(lambda s: "_test_" in s, uttids)))

        # Train/test split exists in expdir
        elif (self.expdir/"train.cfg").exists():
            logger.info(f"Loading dataset splits from spec {self.expdir}/{{train,test}}.cfg")
            train_sections, test_sections = (
                set(read_config(self.expdir/f"{subset}.cfg").get(subset, "datasections").split())
                for subset in ("train", "test")
            )

            def make_filter(sections):
                def _filter(uttid):
                    return any(uttid.startswith(spkr) for spkr in sections)
                return _filter

            train_mask = np.array(list(map(make_filter(train_sections), uttids)))
            valid_mask = np.array(list(map(make_filter(test_sections), uttids)))

        # Random train/test split
        else:
            logger.info("Random train/test split")
            train_ids, valid_ids = train_test_split(uttids, test_size=0.1)
            train_mask = np.array(list(map(lambda s: s in train_ids, uttids)))
            valid_mask = np.array(list(map(lambda s: s in valid_ids, uttids)))

        if options.method in ("10%", "1%"):
            sz = .1 if options.method == "10%" else .01
            train_ids = np.arange(len(features))[train_mask]
            train_ids, _ = train_test_split(train_ids, train_size=sz, stratify=labels[train_mask])
            train_ids = set(train_ids)
            train_mask = [idx in train_ids for idx in np.arange(len(features))]

        train_set = SequenceDataset(features[train_mask], labels[train_mask])
        valid_set = SequenceDataset(features[valid_mask], labels[valid_mask])
        # test_set = SequenceDataset(features[test_mask], labels[test_mask])

        logger.info(f"Dataset loaded: train_size={len(train_set):,} valid_size={len(valid_set):,}")

        return train_set, valid_set

    @staticmethod
    def load_tasks(filename):
        with open(filename) as f:
            spkr = filename.stem

            if spkr == "tasks":
                spkr = filename.parent.name

            uttids, tasks = map(list, zip(*map(
                lambda s: s.split(maxsplit=1),
                map(str.strip, f.readlines()))
            ))

        if not uttids[0].startswith(spkr):
            uttids = list(map(f"{spkr}_{{}}".format, uttids))
        return uttids, tasks

    @staticmethod
    def check_errors(features, labels, action="raise"):
        errors = set(features).union(set(labels)) - set(features).intersection(set(labels))
        msg = f"{len(errors)} mismatches ({len(features)} features and {len(labels)} labels)"

        if not errors:
            pass
        elif action == "raise":
            raise Exception(msg)
        elif action == "warn":
            warning.warn(msg)
        elif action == "ignore":
            features = {k: v for k, v in features.items() if k not in errors}
            labels = {k: v for k, v in labels.items() if k not in errors}
            if not(features and labels):
                raise ValueError("No examples left after removing errors")
        else:
            raise TypeError(f"Unknown action: {action}")

        return features, labels


def load_data(options):

    if options.expdir is None:
        options.expdir = options.outdir

    dataconf = read_config(options.expdir/"database.cfg")
    structure = Structure(options.expdir/"structure.xml")
    coderconf = read_config(options.expdir/"coder.cfg")
    Coder = coder_factory(coderconf.get("coder", "name"))
    coder = Coder(structure, coderconf)

    if options.expdir == options.outdir:
        trainfeats = FeatLoader(options.outdir/"trainfeats").to_dict()
        testfeats = FeatLoader(options.outdir/"testfeats").to_dict()
        with open(options.outdir/"traintasks") as traintasks:
            trainlabels = {
                uttid: coder.encode(read_task(task))
                for uttid, task in map(parse_line, traintasks.readlines())
            }

        with open(options.outdir/"testtasks") as testtasks:
            testlabels = {
                uttid: coder.encode(read_task(task))
                for uttid, task in map(parse_line, testtasks.readlines())
            }

        features = set(trainfeats).union(set(testfeats))
        labels = set(trainlabels).union(set(testlabels))

    else:

        def load_tasks(filename):
            with open(filename) as f:
                spkr = filename.stem

                if spkr == "tasks":
                    spkr = filename.parent.name

                uttids, tasks = map(list, zip(*map(
                    lambda s: s.split(maxsplit=1),
                    map(str.strip, f.readlines()))
                ))

            if not uttids[0].startswith(spkr):
                uttids = list(map(f"{spkr}_{{}}".format, uttids))
            return uttids, tasks

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
        else:
            # import ipdb; ipdb.set_trace()
            features = {k: v for k, v in features.items() if k not in errors}
            labels = {k: v for k, v in labels.items() if k not in errors}
            if not(features and labels):
                raise ValueError("No examples left after removing errors")

    if options.expdir == options.outdir:
        trainuttids = set(trainfeats)
        trainfeats = np.array([trainfeats[uttid] for uttid in trainuttids], dtype="object")
        trainlabels = np.array([trainlabels[uttid] for uttid in trainuttids])
        train_set = SequenceDataset(trainfeats, trainlabels)

        testuttids = set(testfeats)
        testfeats = np.array([testfeats[uttid] for uttid in testuttids], dtype="object")
        testlabels = np.array([testlabels[uttid] for uttid in testuttids])
        valid_set = SequenceDataset(testfeats, testlabels)

        return train_set, valid_set

    uttids = np.array(list(features))
    features = np.array([features[uttid] for uttid in uttids], dtype="object")
    labels = np.array([labels[uttid] for uttid in uttids])

    if options.method == "10-fold":
        return SequenceDataset(features, labels, indices=uttids)

    # 1. Fluent Speech Commands
    if any(subset in uttids[0] for subset in ["train", "valid", "test"]):
        logger.info("Fluent Speech Commands dataset splits")
        train_mask = np.array(list(map(lambda s: "_train_" in s, uttids)))
        valid_mask = np.array(list(map(lambda s: "_valid_" in s, uttids)))
        # test_mask = np.array(list(map(lambda s: "_test_" in s, uttids)))

    # 2. Train/test split exists in expdir
    elif (options.expdir/"train.cfg").exists():
        logger.info(f"Loading dataset splits from spec {options.expdir}/{{train,test}}.cfg")
        train_sections = set(read_config(options.expdir/"train.cfg").get("train", "datasections").split())
        test_sections = set(read_config(options.expdir/"test.cfg").get("test", "datasections").split())

        def make_filter(sections):
            def _filter(uttid):
                return any(uttid.startswith(spkr) for spkr in sections)
            return _filter

        train_mask = np.array(list(map(make_filter(train_sections), uttids)))
        valid_mask = np.array(list(map(make_filter(test_sections), uttids)))

    # 3. Random train/test split
    else:
        logger.info("Random train/test split")
        train_ids, valid_ids = train_test_split(uttids, test_size=0.1)
        train_mask = np.array(list(map(lambda s: s in train_ids, uttids)))
        valid_mask = np.array(list(map(lambda s: s in valid_ids, uttids)))

    if options.method in ("10%", "1%"):
        sz = .1 if options.method == "10%" else .01
        train_ids = np.arange(len(features))[train_mask]
        train_ids, _ = train_test_split(train_ids, train_size=sz, stratify=labels[train_mask])
        train_ids = set(train_ids)
        train_mask = [idx in train_ids for idx in np.arange(len(features))]

    train_set = SequenceDataset(features[train_mask], labels[train_mask])
    valid_set = SequenceDataset(features[valid_mask], labels[valid_mask])
    # test_set = SequenceDataset(features[test_mask], labels[test_mask])

    logger.info(f"Dataset loaded: train_size={len(train_set):,} valid_size={len(valid_set):,}")


    with open(options.outdir/"trainfeats") as featfile, \
            open(options.outdir/"traintasks") as taskfile:
        for uttid in uttids[train_mask]:
            speaker = "_".join(uttid.split("_")[:-1])
            featpath = dataconf.get("speaker")["feats"]
            taskstring = coder.decode(labels[uttid])
            featfile.write(f"{uttid} {featpath}:{uttid}\n")
            taskfile.write(f"{uttid} {taskstring}")


    return train_set, valid_set


def build_model(input_dim, output_dim, options, display=True):
    with open(options.model_config or options.expdir/"model.json") as f:
        config = json.load(f)

    model_name = config.pop("name")
    if model_name not in AVAILABLE_MODELS:
        raise TypeError(f"{model_name} is not one of {list(AVAILABLE_MODELS)}")

    Model = AVAILABLE_MODELS[model_name]
    model = Model(input_dim, output_dim, **config)
    if display:
        display_model(model, print_fn=logger.info)

    return model


def prepare_cv(split, train_index, test_index, dataset, options):
    splitdir = options.outdir/f"split{split}"

    if splitdir.exists():
        shutil.rmtree(splitdir)
    os.makedirs(splitdir)
    logzero.logfile(splitdir/"log")
    logger.info(f"{splitdir} created")

    for fn in ("acquisition.cfg", "coder.cfg", "structure.xml"):
        if not (options.outdir/fn).exists():
            shutil.copy(options.expdir/fn, options.outdir)
        os.symlink(f"../{fn}", splitdir/fn)

    data = tuple()
    for name, index in [("train", train_index), ("test", test_index)]:
        with open(splitdir/f"{name}indices", "w") as f:
            f.writelines(map("{}\n".format, dataset.indices[index]))
        subset = Subset(dataset, index)
        subset.data_collator = dataset.data_collator
        data += (subset,)

    options = deepcopy(options)
    options.outdir = splitdir
    return data, options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", help="JSON with train specific config")
    parser.add_argument("--model-config", type=Path, help="JSON with model specific config")
    parser.add_argument("--device", type=torch.device, default="cuda")
    parser = TrainTestLoader.add_arguments(parser)
    options = parser.parse_args()
    config = {}
    for cfg in options.config:
        with open(cfg) as f:
            config.update(json.load(f))
    options.config = config
    return options


if __name__ == "__main__":

    options = parse_args()
    config = options.config

    os.makedirs(options.outdir, exist_ok=True)
    logzero.logfile(f"{options.outdir}/log")
    logger.info(f"{options.outdir} created")

    loader = TrainTestLoader(options)

    if options.method in ("full", "10%", "1%"):
        train_set, valid_set = loader.train_set, loader.valid_set
        model = build_model(train_set.input_dim, train_set.output_dim, options)
        train(model, train_set, valid_set, options.outdir/"model.pt", device=options.device, **config)
    elif options.method in ("10-fold", "5-fold", "3-fold"):
        extract_uttid = lambda uttid: uttid.split("_")[-1]
        dataset = loader.train_set
        outdir = options.outdir
        nfolds = int(options.method.split("-")[0])
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=1903)
        uttids = np.array(list(set(map(extract_uttid , dataset.indices))))
        for i, (train_index, test_index) in enumerate(kf.split(uttids)):
            train_index = [
                i for i, uttid in enumerate(map(extract_uttid, dataset.indices))
                if uttid in uttids[train_index]
            ]
            test_index = [
                i for i, uttid in enumerate(map(extract_uttid, dataset.indices))
                if uttid in uttids[test_index]
            ]
            logger.info(f"Split #{i}: train size={len(train_index):,} test size={len(test_index):,}")
            (train_set, valid_set), this_opts = prepare_cv(i, train_index, test_index, dataset, options)
            model = build_model(dataset.input_dim, dataset.output_dim, this_opts)
            train(
                model,
                train_set, valid_set,
                this_opts.outdir/"model.pt",
                device=this_opts.device,
                **config
            )

        logzero.logfile(f"{options.outdir}/log")

