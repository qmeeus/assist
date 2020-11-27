import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from assist.acquisition.mlp import Classifier as BaseClassifier


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
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
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

    def __init__(self, features, labels=None):
        self.feature_lengths = [len(feats) for feats in features]
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        tensors = (torch.tensor(self.features[idx]), torch.tensor(self.feature_lengths[idx]))
        if self.labels is not None:
            tensors += (torch.tensor(self.labels[idx]),)
        return tensors

    def __len__(self):
        return len(self.features)


def sort_batch(lengths, *tensors):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    lengths, sort_order = lengths.sort(0, descending=True)
    return (lengths,) + tuple(tensor[sort_order] for tensor in tensors)


def pad_and_sort_batch(batch, sort=False):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    features, lengths, *tensors = list(zip(*batch))
    features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.)
    lengths = torch.stack(lengths, 0)
    tensors = tuple(torch.stack(tensor, 0) for tensor in tensors)
    if sort:
        lengths, features, *tensors = sort_batch(lengths, features, *tensors)
    return (features, lengths) + tuple(tensors)


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config.get("acquisition", "device"))
        return LSTM(
            input_dim=int(self.config.get("acquisition", "input_dim")),
            output_dim=self.n_classes,
            hidden_dim=int(self.config.get("acquisition", "hidden_dim")),
            dropout=float(self.config.get("acquisition", "dropout"))
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

