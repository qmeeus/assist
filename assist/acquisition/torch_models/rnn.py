import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assist.acquisition.torch_models.base import BaseClassifier
from assist.acquisition.torch_models.utils import SequenceDataset, pad_and_sort_batch


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

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

        self.num_directions = 2
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim


    def forward(self, features, lengths, labels=None):
        bs, maxlen, _ = features.size()
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.rnn(packed_sequences)
        out_rnn, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out_rnn = out_rnn.view(bs, maxlen, self.num_directions, self.hidden_dim)
        hidden = self.dropout(torch.cat((
            out_rnn[torch.arange(bs), lengths - 1, 0, :],    # forward
            out_rnn[torch.arange(bs), 0, 1, :]               # backward
        ), dim=-1))

        assert hidden.size() == (bs, self.hidden_dim * self.num_directions)
        logits = self.fc(hidden)

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
            rnn_type=self.config.get("rnn_type", "lstm")
        ).to(self.device)        

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=self.batch_size,
            collate_fn=pad_and_sort_batch
        )

    def prepare_inputs(self, features, labels=None):
        return SequenceDataset(features, labels=labels),

