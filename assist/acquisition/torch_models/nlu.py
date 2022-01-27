import torch
from torch import nn
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from .slu import SLU


class NLU(SLU):

    def forward(self, inputs, input_lengths):
        inputs = inputs.to(self.encoder.device)
        attention_mask = make_non_pad_mask(input_lengths).to(self.encoder.device)
        encodings = self.encoder(inputs, attention_mask=attention_mask)["last_hidden_state"]
        logits = self.classifier(encodings, input_lengths)
        return logits

