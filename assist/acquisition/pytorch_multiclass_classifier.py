import numpy as np
import pandas as pd
import os
import sys
import torch

from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.append(os.environ["HOME"] + "/.local/share/src/dotfiles/scripts/python/data-science")

from tsne_nlp_feats import encode_texts, load_data


REPOS = Path("/esat/spchtemp/scratch/qmeeus/repos")
DATA = REPOS/"espnet/egs/grabo/sti1/data"
TEXT = DATA/"grabo/text.raw"
TARGET = DATA/"grabo_w2v/encoded_target.csv"
MODEL = REPOS/"transformers/examples/language-modeling/output/robbert/checkpoint-21000"


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, features, labels=None):
        logits = self.layers(features)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits
        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class LSTM(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, features, lengths, labels=None):
        bs = features.size(0)
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed_sequences)
        h = torch.transpose(h, 0, 1).contiguous().view(bs, -1)
        h = self.dropout(h)
        logits = self.fc(h)

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits

        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class SequenceDataset:

    def __init__(self, features, target):
        self.feature_lengths = torch.tensor([len(feats) for feats in features])
        self.features = features
        self.target = target

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            self.feature_lengths[idx],
            torch.tensor(self.target[idx])
        )

    def __len__(self):
        return len(self.features)


def sort_batch(sequences, lengths, targets):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = sequences[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, seq_lengths, target_tensor


def pad_and_sort_batch(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_size = len(batch)
    sequences, lengths, target  = list(zip(*batch))
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.)
    max_length = max(lengths)
    return sort_batch(padded_sequences, torch.stack(lengths, 0), torch.stack(target, 0))


def train(model, train_set, test_set, epochs=10, batch_size=128, collate_fn=None):
    optimizer = torch.optim.Adam(model.parameters())
    progress_bar = tqdm(
        total=epochs,
        bar_format="{postfix[1][epoch]}/{postfix[0]} "
        "train={postfix[1][loss]:.4f} test={postfix[1][eval_loss]:.4f}",
        postfix=[epochs, {"epoch": 0, "loss": float('inf'), "eval_loss": float('inf')}]
    )

    with progress_bar:
        for epoch in range(epochs):
            train_loader, test_loader = (
                DataLoader(dataset, shuffle=not(i), batch_size=batch_size, collate_fn=collate_fn)
                for i, dataset in enumerate([train_set, test_set])
            )
            train_loss = train_one_epoch(train_loader, model, optimizer)
            eval_loss = eval_one_epoch(test_loader, model)
            progress_bar.postfix[1].update({
                "epoch": epoch, "loss": train_loss/len(train_set), "eval_loss": eval_loss/len(test_set)
            })
            progress_bar.update()

def evaluate(model, test_set, batch_size=128, collate_fn=None):

    with torch.no_grad():
        logits = []
        labels = []
        for *inputs, yb in DataLoader(
            test_set, shuffle=False, batch_size=batch_size, collate_fn=collate_fn
        ):

            logits.append(model(*inputs))
            labels.append(yb)

        logits, labels = (torch.cat(ys, dim=0) for ys in (logits, labels))
        y_pred = (torch.sigmoid(logits) > .5).long()
        print((y_pred == labels).float().mean().item())
        return logits


def train_one_epoch(dataloader, model, optimizer):
    epoch_loss = 0
    for *inputs, labels in dataloader:
        optimizer.zero_grad()
        loss, _ = model(*inputs, labels=labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def eval_one_epoch(dataloader, model):
    with torch.no_grad():
        epoch_loss = 0
        for *inputs, labels in dataloader:
            epoch_loss += model(*inputs, labels=labels)[0].item()
        return epoch_loss


def train_mlp(textfile, targetfile, nlp_model, epochs=10, batch_size=128):
    texts, target = load_data(textfile, targetfile, target_cols=list(range(1, 32)))
    features = encode_texts(texts, str(nlp_model), "average")
    X_train, X_test, y_train, y_test = train_test_split(features, target.values)
    train_set, test_set = (
        TensorDataset(torch.tensor(X), torch.tensor(y))
        for X, y in [(X_train, y_train), (X_test, y_test)]
    )

    model = MLP(features.shape[1], target.shape[1])
    train(model, train_set, test_set, epochs=epochs, batch_size=batch_size)
    evaluate(model, test_set)


def train_lstm(textfile, targetfile, nlp_model, epochs=10, batch_size=128):
    texts, target = load_data(textfile, targetfile, target_cols=list(range(1, 32)))
    features = encode_texts(texts, str(nlp_model), "none")
    X_train, X_test, y_train, y_test = train_test_split(features, target.values)
    train_set, test_set = (
        SequenceDataset(X, y) for X, y in [(X_train, y_train), (X_test, y_test)]
    )
    model = LSTM(features[0].shape[1], target.shape[1])
    train(model, train_set, test_set, epochs=epochs, batch_size=batch_size, collate_fn=pad_and_sort_batch)
    evaluate(model, test_set, batch_size=batch_size, collate_fn=collate_fn)
