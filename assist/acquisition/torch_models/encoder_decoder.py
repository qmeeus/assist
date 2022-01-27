import json
import torch
import torch.nn as nn
from copy import deepcopy
from espnet.asr.pytorch_backend.asr_init import load_trained_model

from .attention import AttentiveDecoder, AttentiveRecurrentDecoder
from .rnn import RNN, Classifier as BaseClassifier

DECODERS = {
    "att": AttentiveDecoder,
    "att_rnn": AttentiveRecurrentDecoder,
    "rnn": RNN
}


class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, inputs, input_lengths, labels=None):
        encodings, encoding_lengths = self.encoder.encode(inputs, input_lengths)
        return self.decoder(encodings, encoding_lengths, labels)


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config["device"])
        with open(self.config["model_config"]) as f:
            cfg = json.load(f)
        encoder, _ = load_trained_model(cfg.pop("pretrained_encoder"))
        decoder = DECODERS.get(cfg.pop("decoder_type"))(output_dim=self.n_classes, **cfg)        
        return EncoderDecoder(encoder, decoder).to(self.device)

    def get_optimizer(self):
        enc_opt = torch.optim.Adam(self.model.encoder.parameters(), lr=float(self.config["enc_lr"]))
        dec_opt = torch.optim.Adam(self.model.decoder.parameters(), lr=float(self.config["dec_lr"]))
        return [enc_opt, dec_opt]

    def train_one_step(self, inputs, labels, optimizers):
        [optimizer.zero_grad() for optimizer in optimizers]
        *inputs, labels = self._recursive_to(*inputs, labels)
        loss, _ = self.model(*inputs, labels=labels)
        loss.backward()
        [optimizer.step() for optimizer in optimizers]
        return loss.item()
