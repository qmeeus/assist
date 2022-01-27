import torch
import torch.nn as nn
from torch.utils.data import Dataset


__all__ = [
    "SequenceDataset",
    "pad_and_sort_batch",
    "sort_batch",
    "get_attention_mask",
    "count_parameters",
    "display_model",
]


class SequenceDataset(Dataset):

    def __init__(self, features, labels=None, indices=None):
        self.feature_lengths = [len(feats) for feats in features]
        self.features = features
        self.labels = labels
        self.input_dim = features[0].shape[-1]
        self.output_dim = labels[0].shape[-1] if labels is not None else None

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data_collator([self[i] for i in range(*idx.indices(len(self)))])

        tensors = (torch.tensor(self.features[idx]), torch.tensor(self.feature_lengths[idx]))
        if self.labels is not None:
            tensors += (torch.tensor(self.labels[idx]),)

        return tensors

    def __len__(self):
        return len(self.features)

    @staticmethod
    def data_collator(batch):
        return pad_and_sort_batch(batch)


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


def get_attention_mask(lengths):
    maxlen = lengths.max()
    mask = torch.zeros(lengths.size(0), maxlen)
    for i, length in enumerate(lengths):
        mask[i, length:] = 1
    return mask.bool()


def count_parameters(model):
    nparams, ntrainable = 0, 0
    for p in model.parameters():
        nparams += p.numel()
        if p.requires_grad:
            ntrainable += p.numel()
    return nparams, ntrainable


def display_model(model, print_fn=print):
    for line in str(model).split("\n"):
        print_fn(line)

    nparams, ntrainable = count_parameters(model)
    print_fn(f"# parameters: {nparams:,} (trainable: {ntrainable:,})")
