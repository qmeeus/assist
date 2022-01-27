from pathlib import Path
import torch
import kaldiio
import pandas as pd
import numpy as np
from assist.acquisition.torch_models import SequenceDataset
from functools import partial
from itertools import chain
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import log_loss
from dataclasses import dataclass
from assist.tools import read_config, logger
from assist.tasks import read_task, coder_factory, Structure


def load_tasks(taskfile):
    taskfile = Path(taskfile)
    spkr = taskfile.parent.name
    with open(taskfile) as f:
        uttids, tasks = zip(*map(lambda s: s.split(maxsplit=1), map(str.strip, f)))
    return list(zip(map(f"{spkr}_{{}}".format, uttids), tasks))


@dataclass
class Task:
    name:str
    args:dict


# Setup
confdir = Path("config/FluentSpeechCommands/lstm_128")
# outdir = Path("exp/fluent/finetune_gru")
outdir = Path("exp/fluent/finetune_gru_enc_upd_2")
dataconf = read_config(confdir/"database.cfg")
coderconf = read_config(confdir/"coder.cfg")
structure = Structure(confdir/"structure.xml")
Coder = coder_factory(coderconf.get("coder", "name"))
coder = Coder(structure, coderconf)

# Model
encoder = torch.load(outdir/"encoder.pt", map_location="cuda")
decoder = torch.load(outdir/"decoder.pt", map_location="cuda")
for module in (encoder, decoder):
    for line in str(module).split("\n"):
        logger.info(line)
    for p in module.parameters():
        p.requires_grad = False

# Target
speakers = list(dataconf.sections())
taskfiles = list(map(lambda spkr: dataconf[spkr].get("tasks"), speakers))
taskstrings = dict(sum(map(load_tasks, taskfiles), []))
data = pd.DataFrame.from_dict(taskstrings, orient="index")
data.columns = ["taskstring"]
data["subset"] = data.index.map(lambda s: s.split("_")[1])
data["task"] = data["taskstring"].map(read_task).map(lambda t: Task(*t))
data.drop("taskstring", axis=1, inplace=True)
data.sort_index(inplace=True)
train_mask, valid_mask, test_mask = (data.subset == subset for subset in ("train", "valid", "test"))
target = np.stack(list(map(coder.encode, data["task"])), axis=0)

# Features
feat_loader = kaldiio.load_scp(str(confdir.parent/"fbanks.scp"))
features = pd.Series({uttid: feat_loader[uttid] for uttid in feat_loader}).sort_index()
assert (features.index == data.index).all()

# Datasets
get_dataloader = partial(
    DataLoader, batch_size=32, shuffle=False, collate_fn=SequenceDataset.data_collator
)
train, valid, test = (
    SequenceDataset(features[mask], target[mask], data[mask].index)
    for mask in [train_mask, valid_mask, test_mask]
)
subsets = {"train": train, "valid": valid, "test": test}
for name, subset in subsets.items():
    logger.info(f"{name.upper()}: {len(subset)} examples")

# Predict
sigmoid = lambda x: (1 + np.exp(-x)) ** (-1)
error_rate = lambda labels, logits: 1 - (labels == np.rint(sigmoid(logits))).all(-1).mean()
cost_fn = partial(log_loss, normalize=False)
decode = partial(coder.decode, cost=cost_fn)

def predict(dataset):
    loader = get_dataloader(dataset)
    predictions = []
    total_loss = 0
    with torch.no_grad():
        for inputs, input_lengths, labels in tqdm(loader, total=len(dataset)//loader.batch_size):
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            enc, enc_lengths = encoder.encode(inputs, input_lengths)
            loss, preds = decoder(enc, enc_lengths, labels=labels)
            total_loss += float(loss)
            predictions.extend(preds.detach().cpu().tolist())
    return total_loss / len(dataset), np.array(predictions)


for (name, subset), mask in zip(subsets.items(), [train_mask, valid_mask, test_mask]):
    if name == "train": continue
    logger.info("*" * 10 + f"{name.upper()}" + "*" * 10)
    loss, preds = predict(subset)
    raw_score = error_rate(subset.labels, preds)
    decoded = pd.Series(
        list(map(lambda t: Task(*t), map(decode, sigmoid(preds)))),
        index=data[mask].index
    )
    score = (data.loc[mask, "task"] != decoded).astype(int).mean()
    logger.info(f"{name.upper()}: loss={loss:.4f} error rate={raw_score:.4f} decoded={score:.4f}")

