import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assist.acquisition.torch_models.base import BaseClassifier
from assist.acquisition.torch_models.utils import SequenceDataset, pad_and_sort_batch, get_attention_mask


class AttentiveDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, d_model=None, num_heads=8, dropout=0.1, bias=True):
        super(AttentiveDecoder, self).__init__()
        if d_model is not None and input_dim != d_model:
            self.embedding = nn.Linear(input_dim, d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            bias=bias
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, inputs, input_lengths, labels=None, return_attention=False):

        if hasattr(self, "embedding"):
            inputs = self.embedding(inputs)

        keys = inputs.transpose(0, 1).contiguous()
        key_padding_mask = get_attention_mask(input_lengths).to(keys.device)
        linear_combination, energy  = self.attention(
            keys, keys, keys, key_padding_mask=key_padding_mask)

        logits = self.output_layer(self.dropout(linear_combination.sum(0)))

        if labels is None and not return_attention:
            return logits

        output = tuple()

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output += (loss,)

        output += (logits,)

        if return_attention:
            output += (energy,)
        
        return output

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class AttentiveRecurrentDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, d_model, rnn_type="lstm", num_layers=1, rnn_dropout=0.1, num_heads=8, attn_dropout=0.1, attn_bias=True):
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

        self.dropout = nn.Dropout(rnn_dropout)
        self.output_layer = nn.Linear(d_model * 2, output_dim)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, inputs, input_lengths, labels=None, return_attention=False):
        outputs, hidden = self.encoder(inputs)

        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        
        query = hidden.unsqueeze(1).transpose(0, 1).contiguous()
        key = outputs.transpose(0, 1).contiguous()
        linear_combination, energy  = self.attention(query, key, key)
        linear_combination = linear_combination.squeeze(0)
        logits = self.output_layer(self.dropout(linear_combination))

        if labels is None and not return_attention:
            return logits

        output = tuple()
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output += (loss,)

        output += (logits,)

        if return_attention:
            output += (energy,)
        
        return output

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config["device"])

        if self.config["name"] == "att":

            model = AttentiveDecoder(
                input_dim=int(self.config["input_dim"]),
                output_dim=self.n_classes,
                d_model=int(self.config["hidden_dim"]),
                num_heads=int(self.config.get("num_heads", 4)),
                dropout=float(self.config["dropout"]),
                bias=True
            )

        elif self.config["name"] == "att_rnn":

            model = AttentiveRecurrentDecoder(
                input_dim=int(self.config["input_dim"]),
                output_dim=self.n_classes,
                d_model=int(self.config["hidden_dim"]),
                rnn_type=self.config.get("rnn_type", "lstm"),
                num_layers=int(self.config["num_layers"]),
                rnn_dropout=float(self.config["dropout"]), 
                num_heads=int(self.config.get("num_heads", 4)),
                attn_dropout=float(self.config["dropout"])
            )

        else:
            raise NotImplementedError(f"Unknown model: {self.config['name']}")

        return model.to(self.device)

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(
            dataset,
            shuffle=is_train,
            batch_size=self.batch_size,
            collate_fn=pad_and_sort_batch
        )

    def prepare_inputs(self, features, labels=None):
        return SequenceDataset(features, labels=labels),