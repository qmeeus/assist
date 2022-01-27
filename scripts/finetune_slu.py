import argparse
import json
import logzero
import numpy as np
import os
import re
import sys
import torch
from collections import defaultdict
from copy import deepcopy
from functools import partial
from operator import itemgetter
from pathlib import Path
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import trange
from tqdm import tqdm
import assist
from assist.acquisition import model_factory
from assist.acquisition.torch_models import SLU, SLU2
from assist.acquisition.torch_models import NLU
from assist.acquisition.torch_models import RNN
from assist.acquisition.torch_models import display_model
from assist.tasks import Structure
from assist.tasks import read_task
from assist.tools import logger
from assist.tools import read_config
from assist.tools import FeatLoader
from assist.tools import parse_line
from train_lstm_fluent import TrainTestLoader
from train_lstm_fluent import build_model
from train_lstm_fluent import display_model
sys.path.append("/esat/spchtemp/scratch/qmeeus/repos/datasets")
import datasets


class EarlyStopping:

    def __init__(self,
                 max_epochs=100,
                 patience=10,
                 criterion="val_loss",
                 threshold=0,
                 lower_is_better=True):

        logger.info(f"Set early stopping on {criterion} with patience {patience}.")
        self.max_epochs = max_epochs
        self.patience =  patience
        self.criterion = criterion
        self.threshold = threshold
        self.lower_is_better = lower_is_better
        self.initialize()

    def initialize(self):
        self._best_epoch = self.max_epochs
        self._best_score = float("inf") if self.lower_is_better else 0

    def should_stop(self, epoch):
        return (epoch - self._best_epoch) > self.patience or epoch >= self.max_epochs

    def __call__(self, epoch, metrics, model):
        score = metrics[self.criterion]
        l = (-1 if self.lower_is_better else 1)
        if (score - self._best_score) * l > self.threshold:
            self._best_epoch = epoch
            self._best_score = score
            return True
        return False


class Trainer:

    DEFAULT_OPT = ("Adam", {"lr": 1e-2})
    DEFAULT_SCHED = (None, {})

    def __init__(self,
                 model=None,
                 max_epochs=100,
                 warmup_steps=0,
                 batch_size=32,
                 early_stopping=None,
                 encoder_opt=DEFAULT_OPT,
                 decoder_opt=DEFAULT_OPT,
                 encoder_sched=DEFAULT_SCHED,
                 decoder_sched=DEFAULT_SCHED,
                 checkpoint_dir=None,
                 save_interval=1,
                 device="cpu"):

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.early_stopping = early_stopping
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

        self.device = device
        self.model = None
        self.config = {
            "optimizers": {
                "encoder": encoder_opt,
                "decoder": decoder_opt
            },
            "schedulers": {
                "encoder": encoder_sched,
                "decoder": decoder_sched
            }
        }

        self.optimizers = None
        self.schedulers = None

        if model is not None:
            self.set_model(model)

        self.early_stopping = (
            EarlyStopping(max_epochs=max_epochs, **early_stopping)
            if early_stopping is not None else None
        )

    def set_model(self, model):
        self.model = model

    def initialize(self):
        if self.model is None:
            raise TypeError("model is not initialized")
        self.model = self.model.to(self.device)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizers = [
            self.get_optimizer(name, filter(lambda p: p.requires_grad, module.parameters()))
            for name, module in [("encoder", self.model.encoder), ("decoder", self.model.classifier)]
        ]

        self.schedulers = [
            self.get_scheduler(name, optim) if optim is not None else None
            for name, optim in zip(["encoder", "decoder"], self.optimizers)
        ]

    def get_optimizer(self, name, model_parameters):
        classname, config = self.config["optimizers"][name]
        Optimizer = getattr(torch.optim, classname)
        try:
            return Optimizer(model_parameters, **config)
        except ValueError:
            # No parameters
            return None

    def get_scheduler(self, name, optimizer):
        classname, config = self.config["schedulers"][name]
        if classname is None:
            return
        try:
            Scheduler = getattr(assist.tools.lr_scheduler, classname)
        except AttributeError:
            Scheduler = getattr(lr_scheduler, classname)
        return Scheduler(optimizer, **config)

    def train_one_step(self, feats, feats_lengths, labels):
        feats = feats.to(self.device)
        feats_lengths = feats_lengths.to(self.device)
        labels = labels.to(self.device)
        for optimizer in filter(bool, self.optimizers):
            optimizer.zero_grad()
        logits = self.model(feats, feats_lengths)
        loss = self.model.compute_loss(logits, labels)
        loss.backward()

        for i, (optimizer, scheduler) in enumerate(zip(self.optimizers, self.schedulers)):
            if optimizer is not None and (i > 0 or self._epoch >= self.warmup_steps):
                optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                scheduler.step()

        return loss.item()

    @torch.no_grad()
    def eval_one_step(self, feats, feats_lengths, labels):
        feats = feats.to(self.device)
        feats_lengths = feats_lengths.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(feats, feats_lengths)
        metrics = self.get_metrics(logits, labels)
        loss = self.model.compute_loss(logits, labels)
        return loss, metrics

    def train_one_epoch(self, train_set):
        self.model.train()
        log = defaultdict(int)
        train_lr = self.make_dataloader(train_set, self.batch_size, train=True)
        iter_bar = tqdm(train_lr, desc="train", total=len(train_set)//self.batch_size, leave=False)
        for i, batch in enumerate(iter_bar, 1):
            train_loss = self.train_one_step(*batch)
            log["train_loss"] += train_loss
            iter_bar.set_postfix({
                "loss": f"{train_loss/len(batch[0]):.4f}",
                "cumavg": "{:.4f}".format(log["train_loss"]/i/len(batch[0]))
            })
        log["train_loss"] /= len(train_set)
        return log

    @torch.no_grad()
    def evaluate(self, dataset, log=None):
        batch_size = 32
        self.model.eval()
        log = log or defaultdict(int)
        valid_lr = self.make_dataloader(dataset, batch_size, train=False)
        iter_bar = tqdm(valid_lr, desc="eval", total=len(dataset)//batch_size, leave=False)
        logger.debug(f"Evaluate {len(dataset)} examples")
        for batch in iter_bar:
            val_loss, metrics = self.eval_one_step(*batch)
            log["val_loss"] += float(val_loss)
            for metric, value in metrics.items():
                log[metric] += value
            iter_bar.set_postfix(dict(
                val_loss=f"{val_loss/batch_size:.3f}",
                **{k: f"{v:.3f}" for k, v in self.accumulate_metrics(metrics).items()}
            ))
        log["val_loss"] /= len(dataset)
        return log

    def train(self, train_set, valid_set, test_set=None):
        self.initialize()

        epoch_bar = trange(self.max_epochs, desc="epochs")
        logger.debug(f"Training for {self.max_epochs} epochs on {len(train_set)} examples")
        history = defaultdict(list)
        for self._epoch in epoch_bar:
            log = self.train_one_epoch(train_set)
            log = self.evaluate(valid_set, log)
            [history[key].append(log) for key in log]

            epoch_log = self.on_epoch_end(log, progress_bar=epoch_bar)
            if self.early_stopping and self.early_stopping.should_stop(self._epoch):
                logger.info("Early stopping reached")
                break
            for scheduler in self.schedulers:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(log["val_loss"])

        with open(f"{self.checkpoint_dir}/history.json", "w") as f:
            json.dump(history, f)

        if test_set is not None:
            log = self.evaluate(test_set)
            with open(f"{self.checkpoint_dir}/results.json", "w") as f:
                json.dump(self.accumulate_metrics(log), f, indent=4)

    def on_epoch_end(self, log, progress_bar=None):
        log = self.accumulate_metrics(log)
        if progress_bar:
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in log.items()})
        else:
            logger.debug(f"[{self._epoch+1}] " + " ".join(f"[{k}={v:.4f}]" for k, v in log.items()))
        if self._epoch + 1 % self.save_interval == 0:
            self.save_checkpoint(f"{self._epoch:04d}")
        best = False
        if self.early_stopping and self.early_stopping(self._epoch, log, self.model):
            self.save_checkpoint("best")
        return log

    @torch.no_grad()
    def get_metrics(self, logits, labels):
        if labels.ndim == 1:
            predictions = torch.argmax(logits, -1, keepdim=True)
            labels = labels.unsqueeze(-1)
        else:
            probs = torch.sigmoid(logits)
            predictions = (probs >= .5).long()
        correct = (predictions == labels).detach().cpu().numpy()
        return {
            "correct": int(correct.sum()),
            "all_correct": int(correct.all(-1).sum()),
            "num_examples": len(logits),
            "num_elements": logits.numel()
        }

    def accumulate_metrics(self, metrics):
        output = {
            "accuracy": metrics["correct"] / metrics["num_elements"],
            "error_rate": 1 - metrics["all_correct"] / metrics["num_examples"],
        }
        if "train_loss" in metrics:
            output["train_loss"] = metrics["train_loss"]
        if "val_loss" in metrics:
            output["val_loss"] = metrics["val_loss"]
        return output

    def save_checkpoint(self, suffix):
        path = f"{self.checkpoint_dir}/ckpt-{suffix}.pt"
        logger.debug(f"Saving model to {path}")
        torch.save(self.model, path)

    @staticmethod
    def make_dataloader(dataset, batch_size, train=True):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=dataset.data_collator
        )


class CrossValidation:

    RND = 1903

    @staticmethod
    def parse_args(parser):
        parser.add_argument_group("Cross Validation")
        parser.add_argument("--kfold", type=int, default=0, help="zero: no cv")
        parser.add_argument("--split-method", type=str, default=None, help="How to split data")
        parser.add_argument("--data-dir", type=Path, default=None, help="Precomputed split location")
        return parser

    @classmethod
    def from_args(cls, trainer, model, savedir, options):
        return cls(
            trainer,
            model,
            savedir,
            k=options.kfold,
            method=options.split_method,
            data_dir=options.data_dir
        )

    def __init__(self, trainer, model, savedir, k=10, method=None, data_dir=None):
        self.trainer = trainer
        self.savedir = savedir
        self.initial_model_path = f"{self.savedir}/inital_model.pt"
        os.makedirs(self.savedir, exist_ok=True)
        self.k = k
        self.method = method
        self.data_dir = data_dir

        self.trainer.set_model(model)
        torch.save(model, self.initial_model_path)

    def initialize_trainer(self, fold):
        self.trainer.checkpoint_dir = outdir = f"{self.savedir}/cv{fold}"
        self.trainer.model = torch.load(self.initial_model_path)
        if self.data_dir is not None:
            path = self.data_dir/f"split{fold}/model.pt"
            logger.info(f"Loading pretrained classifier from {path}")
            self.trainer.model.classifier = torch.load(path)
        self.trainer.early_stopping.initialize()
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def split_dataset(self, dataset, train_index, test_index):
        return tuple(datasets.Subset(dataset, index) for index in (train_index, test_index))

    def _split_speakers(self, dataset):
        # Assume index of the form spkr_uttid
        indices = dataset.indices
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=self.RND)
        index_pairs = map(self.split_uttid, indices)
        speakers = np.array(list(set(map(itemgetter(0), index_pairs))))
        if len(speakers) < self.k:
            logger.warn(f"Set number of fold equal to number of speakers ({len(speakers)})")
            self.k = len(speakers)
            kfold.n_splits = self.k

        for split in kfold.split(speakers):
            yield tuple(
                np.array([
                    i for i, idx in enumerate(indices)
                    if self.split_uttid(idx)[0] in speakers[subset]
                ]) for subset in split
            )

    def _split_utterances(self, dataset):
        indices = dataset.indices
        kfold = KFold(n_splits=self.k, shuffle=True, random_state=self.RND)
        index_pairs = map(self.split_uttid, indices)
        sentences = np.array(list(set(map(itemgetter(1), index_pairs))))
        for split in kfold.split(sentences):
            yield tuple(
                np.array([
                    i for i, idx in enumerate(indices)
                    if self.split_uttid(idx)[1] in sentences[subset]
                ]) for subset in split
            )


    def get_splits(self, dataset):
        if self.method is None:
            kfold = KFold(n_splits=self.k, shuffle=True, random_state=self.RND)
            yield from kfold.split(np.arange(len(dataset)))
            return

        assert dataset.indices is not None, "Missing indices"
        if self.method == "load":
            yield from self._load_splits(dataset)
        elif self.method == "speakers":
            yield from self._split_speakers(dataset)
        elif self.method == "utterances":
            yield from self._split_speakers(dataset)
        else:
            raise NotImplementedError(self.method)

    def _load_splits(self, dataset):
        indices = dataset.indices
        for splitdir in self.data_dir.glob("split*"):
            with open(splitdir/"trainfeats") as f:
                trainindex = set(map(lambda line: parse_line(line)[0], f))
                trainmask = np.array([k in trainindex for k in indices])
                trainindex = np.arange(len(indices))[trainmask]
            with open(splitdir/"testfeats") as f:
                testindex = set(map(lambda line: parse_line(line)[0], f))
                testmask = np.array([k in testindex for k in indices])
                testindex = np.arange(len(indices))[testmask]
                assert (trainmask == ~testmask).all()
            yield trainindex, testindex

    @staticmethod
    def split_uttid(uttid):
        return uttid.split("_", maxsplit=1)

    def run(self, dataset):

        detailed_log = []
        for i, (train_index, test_index) in enumerate(self.get_splits(dataset)):
            outdir = self.initialize_trainer(i)
            logzero.logfile(f"{outdir}/log")

            indices = (dataset.indices[train_index], dataset.indices[test_index])
            train_set, test_set = self.split_dataset(dataset, train_index, test_index)
            datasets.Dataset.save_splits(outdir, {"train": train_set, "test": test_set})
            trainer.train(train_set, test_set, test_set)
            log = trainer.evaluate(test_set)
            detailed_log.append(log)
            # if i == 2:  # DEBUG
            #     break

        logzero.logfile(f"{self.savedir}/log")
        with open(f"{self.savedir}/log", "w") as f:
            json.dump(detailed_log, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, nargs="+", help="JSON with train specific config")
    parser.add_argument("--model-config", type=Path, help="JSON with model specific config")
    parser.add_argument("--device", type=torch.device, default="cuda")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--nlu", action="store_true", help="Use BERT instead of Speech Encoder")
    parser.add_argument("--outdir", type=Path, help="exp/fluent/train_lstm_128")
    parser.add_argument("--split-sizes", type=str, default=None, help="Size of each split, eg. 7,1,2")
    parser.add_argument("--freeze-modules", type=str, nargs="+", help="Modules to freeze")
    parser.add_argument("--dataset-size", type=float, default=1., help="Limit the size of the dataset")
    parser = datasets.Dataset.parse_args(parser)
    parser = CrossValidation.parse_args(parser)
    # TODO: Add parse_args function in Model
    options = parser.parse_args()
    config = {}
    for cfg in options.config:
        with open(cfg) as f:
            config.update(json.load(f))
    options.config = config
    return options


if __name__ == "__main__":

    options = parse_args()
    os.makedirs(options.outdir, exist_ok=True)
    logzero.logfile(f"{options.outdir}/log")

    logger.info("Loading inputs and targets")
    data = datasets.Dataset.from_args(options)

    # TODO: Move to model as load, load_encoder and load_classifier
    logger.info("Loading models")
    if options.nlu:
        from transformers import AutoModel, AutoTokenizer
        Model = NLU
        encoder = AutoModel.from_pretrained(options.encoder)
        tokenizer = AutoTokenizer.from_pretrained(options.encoder)
        encoding_dim = encoder.config.hidden_size
        setattr(data, "tokenizer", tokenizer)
    else:
        Model = SLU2
        encoder = torch.load(options.encoder)
        # FIXME: hack
        encoding_dim = (
            encoder.decoder.teacher_proj.out_features
            if "e2e_sti_transformer" in str(encoder.__class__)
            else encoder.adim
        )
        encoding_dim = encoder.adim
        encoder = encoder.encoder

    for p in encoder.parameters():
        p.requires_grad = True

    # TODO: Move to Model
    if options.freeze_modules:
        from assist.tools.torch_utils import freeze_modules

        encoder = freeze_modules(encoder, options.freeze_modules)

    train_set = data("train", p=options.dataset_size)

    if options.classifier is not None:
        classifier = torch.load(options.classifier)
    elif options.model_config is not None:
        classifier = build_model(encoding_dim, train_set.output_dim, options, display=False)
    else:
        raise ValueError("model_config is required if classifier is not given")

    Loss = nn.BCEWithLogitsLoss if data.output_key == "tasks" else nn.CrossEntropyLoss
    model = Model(encoder, classifier, Loss(reduction="sum"))
    display_model(model, print_fn=logger.info)

    # TODO: add parse_args and from_args functions to trainer
    logger.info("Training starts...")
    trainer = Trainer(
        checkpoint_dir=options.outdir,
        device=options.device,
        **options.config
    )

    if options.kfold:
        CrossValidation.from_args(trainer, model, options.outdir, options).run(train_set)
        # CrossValidation(
        #     trainer,
        #     model,
        #     options.outdir,
        #     k=options.kfold,
        #     method=options.split_method
        # ).run(train_set)
    else:
        if options.split_sizes and re.match("\d+/\d+/\d+", options.split_sizes):
            sizes = list(map(float, options.split_sizes.split("/")))
            sizes = [s / sum(sizes) for s in sizes]
            train_set, valid_set, test_set = data.split("train", *sizes)
            data.save_splits(options.outdir, {"train": train_set, "valid": valid_set, "test": test_set})
        else:
            valid_set = data("valid")
            if data.has_subset("test"):
                test_set = data("test")
            else:
                logger.warn("Evaluating on the validation set. Add test set to data configuration.")
                test_set = valid_set

        trainer.set_model(model)
        trainer.train(train_set, valid_set, test_set)

