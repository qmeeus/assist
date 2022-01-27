import argparse
import h5py
import json
import numpy as np
import os
import torch
import torch.nn as nn
import warnings

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn import metrics
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from torch.utils.data import Dataset, DataLoader

from assist.tasks import Structure, read_task, coder_factory
from assist.tools import read_config


class RNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1, dropout=.1, rnn_type="lstm"):
        super(RNN, self).__init__()

        RNNClass = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM

        self.rnn = RNNClass(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0, # if num_layers == 1 else dropout,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim


    def forward(self, inputs, input_lengths, labels=None):
        bs, maxlen, _ = inputs.size()
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.rnn(packed_sequences)
        out_rnn, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out_rnn = out_rnn.view(bs, maxlen, self.num_directions, self.hidden_dim)
        hidden = self.dropout(torch.cat((
            out_rnn[torch.arange(bs), input_lengths - 1, 0, :],    # forward
            out_rnn[torch.arange(bs), 0, 1, :]               # backward
        ), dim=-1))

        assert hidden.size() == (bs, self.hidden_dim * self.num_directions)
        logits = self.fc(hidden)

        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())



class AttentiveRecurrentDecoder(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model,
                 rnn_type="lstm",
                 num_layers=1,
                 rnn_dropout=0.1,
                 num_heads=8,
                 attn_dropout=0.1,
                 attn_bias=True):

        super(AttentiveRecurrentDecoder, self).__init__()
        RNNClass = getattr(nn, rnn_type.upper())
        self.encoder = RNNClass(
            input_dim,
            d_model,
            num_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=attn_bias
        )

        self.output_layer = nn.Linear(d_model * 2, output_dim)

    def forward(self, inputs, input_lengths, labels=None, return_attention=False):
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.encoder(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)

        query = hidden.unsqueeze(1).transpose(0, 1).contiguous()
        key = outputs.transpose(0, 1).contiguous()
        linear_combination, energy  = self.attention(query, key, key)
        linear_combination = linear_combination.squeeze(0)
        logits = self.output_layer(linear_combination)

        if return_attention:
            return logits, energy

        return logits


AVAILABLE_MODELS = {
    # "att": AttentiveDecoder,
    "att_rnn": AttentiveRecurrentDecoder,
    "rnn": RNN
}


class SequenceDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.feature_lengths = [len(feats) for feats in self.features]
        self.labels = labels

        self.input_dim = self.features[0].shape[-1]
        self.output_dim = self.labels[0].shape[-1]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.data_collator([self[i] for i in range(*index.indices(len(self)))])

        inputs = torch.tensor(self.features[index])
        input_lengths = torch.tensor(self.feature_lengths[index])
        labels = torch.tensor(self.labels[index])
        return inputs, input_lengths, labels

    def __len__(self):
        return len(self.features)

    @staticmethod
    def data_collator(batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        # batch, _ = map(list, zip(*batch))
        features, lengths, labels = map(list, zip(*batch))
        features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.)
        lengths = torch.stack(lengths, 0)
        labels = torch.stack(labels, 0)
        # See https://skorch.readthedocs.io/en/latest/user/neuralnet.html#multiple-input-arguments
        return {"inputs": features, "input_lengths": lengths}, labels


def error_rate(y_true, y_pred):
    y_pred = y_pred[:, 1, :] > .5
    return 1 - (y_true == y_pred).all(-1).mean()


def accuracy(y_true, y_pred):
    y_pred = y_pred[:, 1, :] > .5
    return (y_true == y_pred).mean()


def load_tasks(filename):
    with open(filename) as f:
        spkr = filename.stem
        if spkr == "tasks":
            spkr = filename.parent.name
        uttids, tasks = map(list, zip(*map(lambda s: s.split(maxsplit=1), map(str.strip, f.readlines()))))
        return list(map(f"{spkr}_{{}}".format, uttids)), tasks


def train(dataset):

    ids = np.arange(len(dataset))
    train_ids = np.random.choice(ids, int(len(dataset) * .8), replace=False)
    test_ids = np.array(list(filter(lambda i: i not in train_ids, ids)))
    train_data = torch.utils.data.Subset(dataset, train_ids)
    valid_data = torch.utils.data.Subset(dataset, test_ids)

    with open(options.configdir/"model.json") as f:
        model_config = json.load(f)

    with open(options.configdir/"train.json") as f:
        train_config = json.load(f)

    model = AttentiveRecurrentDecoder(
        input_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        **model_config
    )

    loss_fct = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(train_config["max_epochs"]):
        train_lr, valid_lr = (DataLoader(
            data,
            batch_size=train_config["batch_size"],
            shuffle=not(i),
            collate_fn=dataset.data_collator
        ) for i, data in enumerate([train_data, valid_data]))

        train_loss = 0
        for Xi, yi in train_lr:
            optim.zero_grad()
            logits = model(**Xi)
            loss = loss_fct(logits, yi)
            optim.step()
            train_loss += loss.item() * len(yi)

        with torch.no_grad():
            predictions = None
            labels = None
            valid_loss = 0
            for Xi, yi in valid_lr:
                logits = model(**Xi)
                loss = loss_fct(logits, yi)
                valid_loss += loss.item() * len(yi)
                if predictions is None:
                    predictions = logits
                    labels = yi
                else:
                    predictions = torch.cat((predictions, logits), 0)
                    labels = torch.cat((labels, yi), 0)

            val_loss = loss_fct(predictions, labels)
            soft_preds = torch.sigmoid(predictions)
            hard_preds = (soft_preds > .5).long()
            acc = (hard_preds == labels).float().mean()
            er = (hard_preds == labels).all(-1).float().mean()

            print(
                f"Epoch {epoch}: train loss={train_loss / len(train_data):.3f}"
                f" valid loss={valid_loss / len(valid_data):.3f}"
                f" error rate={er:.3f} accuracy={acc:.3f}"
            )


def load_data(options):

    structure = Structure(options.configdir/"structure.xml")
    coderconf = read_config(options.configdir/"coder.cfg")
    Coder = coder_factory(coderconf.get("coder", "name"))
    coder = Coder(structure, coderconf)

    dataconf = read_config(options.configdir/"database.cfg")

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

    uttids = list(features)
    features = [features[uttid] for uttid in uttids]
    labels = [labels[uttid] for uttid in uttids]
    return SequenceDataset(features, labels)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", type=Path, help="Path to output dir")
    parser.add_argument("configdir", type=Path, help="Path to config dir")
    parser.add_argument("--model_config", type=Path, help="JSON file with model specific config")
    # parser.add_argument(
    #     "--model", default="att_rnn", choices=list(AVAILABLE_MODELS),
    #     help="One of {}".format(", ".join(AVAILABLE_MODELS)))
    parser.add_argument("--device", type=torch.device, default="cuda")
    parser.add_argument("--errors", choices=["ignore", "raise", "warn"], default="raise")
    options = parser.parse_args()

    os.makedirs(options.expdir, exist_ok=True)
    dataset = load_data(options)

    with open(options.model_config or options.configdir/"model.json") as f:
        model_config = json.load(f)
        model_name = model_config.pop("name")
        if model_name not in AVAILABLE_MODELS:
            raise TypeError(f"{model_name} is not one of {list(AVAILABLE_MODELS)}")
        
        Model = AVAILABLE_MODELS[model_name]
        config = {f"module__{name}": value for name, value in model_config.items()}
        
    with open(options.configdir/"train.json") as f:
        config.update(json.load(f))

    net = NeuralNet(
        module=Model,
        module__input_dim=dataset.input_dim,
        module__output_dim=dataset.output_dim,
        criterion=nn.BCEWithLogitsLoss,
        iterator_train__collate_fn=dataset.data_collator,
        iterator_valid__collate_fn=dataset.data_collator,
        callbacks=[
            EpochScoring(scoring=metrics.make_scorer(error_rate), lower_is_better=True),
            EpochScoring(scoring=metrics.make_scorer(accuracy), lower_is_better=False),
            EarlyStopping(monitor="valid_loss", patience=5),
            LRScheduler(policy="ReduceLROnPlateau", patience=3)
        ],
        device=options.device,
        **config
    )

    net.fit(dataset)
    logits = net.forward(dataset)
    probits = torch.sigmoid(logits)
    preds = (probits > .5).long().numpy()
    labels = np.stack(dataset.labels)
    correct = (preds == labels)
    positive = (labels == 1)
    pred_pos = (preds == 1)
    TP = correct & positive
    FP = (~correct) & (~positive)
    TN = correct & (~positive)
    FN = (~correct) & positive
    precision = TP.sum()/pred_pos.sum()
    recall = TP.sum()/positive.sum()
    f1_score = 2 * precision * recall / (precision + recall)
    acc = correct.mean()
    erate = correct.all(-1).mean()
    print(" ".join(f"{k}={v.mean():.4f}" for k, v in {"TP": TP, "FP": FP, "TN": TN, "FN": FN}.items()))
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1_score:.4f} acc={acc:.4f} er={erate:.4f}")
    print(metrics.classification_report(y_true=labels, y_pred=preds))





