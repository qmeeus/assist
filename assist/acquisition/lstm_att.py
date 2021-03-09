import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from assist.tools import logger
from .lstm import BaseClassifier, DataLoader, SequenceDataset, pad_and_sort_batch

RNNS = ['lstm', 'gru']

class Encoder(nn.Module):

    def __init__(
        self, 
        embedding_dim, 
        hidden_dim, 
        num_layers=1, 
        dropout=0., 
        bidirectional=True, 
        rnn_type='GRU'):

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_cell = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cell(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, attention, hidden_dim, num_classes):
    
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.attention = attention
        if encoder.bidirectional:
            hidden_dim *= 2
        self.decoder = nn.Linear(hidden_dim, num_classes)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

        size = 0
        for p in self.parameters():
            size += p.nelement()

        logger.info(f'Total param size: {size:,}')



    def forward(self, input, input_lengths, labels=None, return_attention=False):
        outputs, hidden = self.encoder(input)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)
        # import ipdb; ipdb.set_trace()
        # energy, linear_combination = self.attention(hidden, outputs, outputs)
        query = hidden.unsqueeze(1).transpose(0, 1).contiguous()
        key = outputs.transpose(0, 1).contiguous()
        linear_combination, energy  = self.attention(query, key, key)
        linear_combination = linear_combination.squeeze(0)
        logits = self.decoder(linear_combination)

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

        hidden_dim = int(self.config["hidden_dim"])
        dropout = float(self.config["dropout"])

        encoder = Encoder(
            int(self.config["input_dim"]),
            hidden_dim, 
            num_layers=int(self.config["num_layers"]), 
            dropout=dropout, 
            bidirectional=True, 
            rnn_type=self.config.get("rnn_type", "lstm")
        )

        # attention = Attention(hidden_dim, hidden_dim, hidden_dim)
        mha = nn.MultiheadAttention(hidden_dim * 2, 4, dropout, bias=True)
        model = EncoderDecoder(encoder, mha, hidden_dim, self.n_classes)
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