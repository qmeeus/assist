import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assist.acquisition.torch_models.base import BaseClassifier
from assist.acquisition.torch_models.utils import SequenceDataset, pad_and_sort_batch


class RNNLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, rnn_type="lstm", activation=nn.LeakyReLU(0.1), batch_norm=False, dropout=.1):

        super(RNNLayer, self).__init__()

        RNNClass = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM

        self.rnn = RNNClass(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        if activation is not None:
            self.activation = activation

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim

    def forward(self, features, lengths):
        bs, maxlen, _ = features.size()

        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.rnn(packed_sequences)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if hasattr(self, "batch_norm"):
            h = self.batch_norm(h.transpose(1, 2)).transpose(1, 2)

        if hasattr(self, "activation"):        
            h = self.activation(h)
        
        if hasattr(self, "dropout"):
            h = self.dropout(h)

        return h


class RNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1, dropout=.1, rnn_type="lstm", batch_norm=False):
        super(RNN, self).__init__()

        self.rnn_layers = nn.ModuleList([
            RNNLayer(
                input_dim=hidden_dim * 2 if i else input_dim, 
                hidden_dim=hidden_dim, 
                rnn_type=rnn_type,
                batch_norm=batch_norm,
                dropout=dropout
            ) for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, features, lengths, labels=None):
        bs, maxlen, _ = features.size()

        # RNN forward
        rnn_out = features
        for layer in self.rnn_layers:
            rnn_out = layer(rnn_out, lengths)

        # Extract last state
        rnn_out = rnn_out.view(bs, maxlen, 2, self.hidden_dim)
        fw = rnn_out[torch.arange(bs), lengths - 1, 0, :]
        bw = rnn_out[torch.arange(bs), 0, 1, :]
        rnn_out = torch.cat([fw, bw], dim=-1)
        assert rnn_out.size() == (bs, self.hidden_dim * self.num_directions)

        # Output layer
        logits = self.fc(rnn_out)

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits

        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config["device"])
        return RNN(
            input_dim=int(self.config["input_dim"]),
            output_dim=self.n_classes,
            hidden_dim=int(self.config["hidden_dim"]),
            num_layers=int(self.config["num_layers"]),
            dropout=float(self.config["dropout"]),
            rnn_type=self.config.get("rnn_type", "lstm"),
            batch_norm=self.config.get("batch_norm", None) == "true"
        ).to(self.device)

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=self.batch_size,
            collate_fn=pad_and_sort_batch,
            drop_last=is_train
        )

    def prepare_inputs(self, features, labels=None):
        return SequenceDataset(features, labels=labels),

