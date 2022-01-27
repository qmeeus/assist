import argparse
import h5py
import json
import kaldiio
import numpy as np
import os
import os.path as p
import torch
import warnings

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from espnet.bin.asr_train import get_parser
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.utils.dynamic_import import dynamic_import

from assist.acquisition.torch_models import (
    SequenceDataset,
    AttentiveDecoder,
    AttentiveRecurrentDecoder,
    RNN,
    display_model
)

from assist.tasks import Structure, read_task
from assist.tools import read_config, logger
from assist.tasks import coder_factory


AVAILABLE_MODELS = {
    "att": AttentiveDecoder,
    "att_rnn": AttentiveRecurrentDecoder,
    "rnn": RNN
}

EGS_ROOT = "/esat/spchtemp/scratch/qmeeus/repos/espnet-stable/egs/cgn/asr1"
MODEL_NAME = "model.val5.avg.best"
N_FILTERS, ENCODING_SIZE, VOC_SIZE = 80, 768, 5005


def load_dict(filename):
    with open(filename, "rb") as f:
            dictionary = f.readlines()
    char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
    char_list.insert(0, "<blank>")
    char_list.append("<eos>")
    return char_list


def load_args(model_dir):
    parser = get_parser()
    with open(f"{model_dir}/train.log") as f:
        cmd_args = f.readlines()[0].split()[2:]
    for i in range(len(cmd_args)):
        if "/" in cmd_args[i]:
            cmd_args[i] = f"{model_dir.parents[1]}/{cmd_args[i]}"

    args, _ = parser.parse_known_args(cmd_args)
    args.char_list = load_dict(args.dict) if args.dict is not None else None
    args.resume = str(Path(args.resume).parent/MODEL_NAME)
    return args


def load_data(options):

    logger.info("Loading data")
    structure = Structure(options.expdir/"structure.xml")
    coderconf = read_config(options.expdir/"coder.cfg")
    Coder = coder_factory(coderconf.get("coder", "name"))
    coder = Coder(structure, coderconf)
    featconf = read_config(options.expdir/"features.cfg")
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

    feat_loader = kaldiio.load_scp(featconf["features"].get("fbanks"))
    features = {uttid: feat_loader[uttid] for uttid in feat_loader}

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
            logger.warn(msg)

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
        train_ids, valid_ids = train_test_split(uttids, test_size=0.1)
        train_mask = np.array(list(map(lambda s: s in train_ids, uttids)))
        valid_mask = np.array(list(map(lambda s: s in valid_ids, uttids)))

    train_set = SequenceDataset(features[train_mask], labels[train_mask], uttids[train_mask])
    valid_set = SequenceDataset(features[valid_mask], labels[valid_mask], uttids[valid_mask])
    # test_set = SequenceDataset(features[test_mask], labels[test_mask], uttids[test_mask])

    return train_set, valid_set


def build_encoder(model_dir, freeze=None):
    options = load_args(model_dir)
    model, train_args = load_trained_model(options.resume)
    model.teacher_model = None
    if freeze:
        for m in freeze:
            logger.info(f"Freeze {m} in encoder")
            for p in getattr(model, m).parameters():
                p.requires_grad = False
    display_model(model, logger.info)
    logger.debug(train_args)
    return model


def build_decoder(input_dim, output_dim, options):
    with open(options.model_config or options.expdir/"model.json") as f:
        config = json.load(f)

    model_name = config.pop("name")
    if model_name not in AVAILABLE_MODELS:
        raise TypeError(f"{model_name} is not one of {list(AVAILABLE_MODELS)}")
    Model = AVAILABLE_MODELS[model_name]

    model = Model(input_dim, output_dim, **config)
    display_model(model, logger.info)

    return model


def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


def compute_error_rate(labels, logits):
    assert labels.shape == logits.shape
    predictions = np.rint(sigmoid(logits))
    return 1 - (labels == predictions).all(-1).mean()


class Trainer:

    def __init__(self, encoder, decoder):

        self.encoder = encoder
        self.decoder = decoder

    def train_loop(self, ):
        train_loss = 0
        self.encoder.train()
        self.decoder.train()
        predictions, target = None, None
        for inputs, input_lengths, labels in tqdm(train_lr, total=len(train_set)//train_lr.batch_size):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            encodings, encoding_lengths = encoder.encode(inputs, input_lengths)
            loss, preds = decoder(encodings, encoding_lengths, labels=labels)
            loss.backward()
            logger.debug(f"loss={loss.item()}")
            encoder_optimizer.step()
            decoder_optimizer.step()
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
        return {"loss": train_loss, "error_rate": train_error_rate}


def train(
    encoder, decoder,
    train_set, valid_set,
    enc_ckpt, dec_ckpt,
    max_epochs=20,
    batch_size=64,
    enc_lr=1e-5, dec_lr=1e-2,
    es_patience=5,
    es_criterion="val_loss",
    es_lower_is_better=True,
    device="cuda",
    enc_update_interval=1
):

    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=enc_lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=dec_lr)

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
            torch.save(encoder, enc_ckpt)
            torch.save(decoder, dec_ckpt)


    while True:
        train_lr, val_lr = (DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not i,
            collate_fn=dataset.data_collator
        ) for i, dataset in enumerate([train_set, valid_set]))

        train_loss, val_loss = 0, 0
        encoder.train()
        decoder.train()
        predictions, target = None, None
        for i, (inputs, input_lengths, labels) in enumerate(tqdm(train_lr, total=len(train_set)//train_lr.batch_size), 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            encodings, encoding_lengths = encoder.encode(inputs, input_lengths)
            loss, preds = decoder(encodings, encoding_lengths, labels=labels)
            loss.backward()
            logger.debug(f"loss={loss.item()}")
            if i % enc_update_interval == 0:
                encoder_optimizer.step()
            decoder_optimizer.step()
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
            encoder.eval()
            decoder.eval()
            for inputs, input_lengths, labels in val_lr:
                inputs = inputs.to(device)
                labels = labels.to(device)
                encodings, encoding_lengths = encoder.encode(inputs, input_lengths)
                loss, preds = decoder(encodings, encoding_lengths, labels=labels)
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
        logger.info(f"Epoch {epoch} [ TRAIN {train_loss:.4f} {train_error_rate:.4f} ] "
                    f"[ VALID {val_loss:.4f} {val_error_rate:.4f} ]")

        check_early_stopping(epoch, val_loss, val_error_rate)

        if should_stop(epoch):
            break

    logger.info(f"Training completed in {time() - t0:.2f} s. "
                f"Best epoch: {best_epoch} Loss: {best_loss:.3f} ER: {best_error_rate:.4%}")


if __name__ == "__main__":

    split_mods = lambda s: s.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path, help="exp/fluent/train_lstm_128")
    parser.add_argument("expdir", type=Path, help="config/FluentSpeechCommands/lstm_128")
    parser.add_argument("--pretrained-model", type=Path,
                        help="path to pretrained model dir (must contain train.log)")
    parser.add_argument("--model-config", type=Path, help="JSON with model specific config")
    parser.add_argument("--config", type=Path, nargs="+", help="JSON with train specific config")
    parser.add_argument("--errors", choices=["ignore", "raise", "warn"], default="raise")
    parser.add_argument("--device", type=torch.device, default="cuda")
    parser.add_argument("--freeze", type=split_mods, help="modules to freeze, separated with commas")
    parser.add_argument("--encoder-update-interval", type=int, default=1, help="Update encoder every X iterations")
    options = parser.parse_args()

    config = {}
    for cfg in options.config:
        with open(cfg) as f:
            config.update(json.load(f))

    enc_ckpt = options.outdir/"encoder.pt"
    dec_ckpt = options.outdir/"decoder.pt"
    os.makedirs(options.outdir, exist_ok=True)
    train_set, valid_set = load_data(options)

    encoder = build_encoder(options.pretrained_model, options.freeze)
    decoder = build_decoder(ENCODING_SIZE, train_set.output_dim, options)

    train(
        encoder, decoder,
        train_set, valid_set,
        enc_ckpt, dec_ckpt,
        device=options.device,
        enc_update_interval=options.encoder_update_interval,
        **config
    )


