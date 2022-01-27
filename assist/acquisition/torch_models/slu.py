from copy import deepcopy
import torch
from torch import nn
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class SLU(nn.Module):

    def __init__(self, encoder, classifier, loss=nn.BCEWithLogitsLoss(reduction="sum")):
        super(SLU, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.loss = loss

    def forward(self, feats, feats_lengths):

        src_mask = make_non_pad_mask(feats_lengths.tolist()).to(feats.device).unsqueeze(-2)
        src_mask = src_mask.to(feats.device)
        h, hs_mask = self.encoder.encoder(feats, src_mask)
        ys_in_pad = self.encoder.init_decoder(h, .9)
        ys_mask = target_mask(ys_in_pad, ignore_id=self.encoder.eos).to(ys_in_pad.device)
        _, mask, encodings = self.encoder.decoder(ys_in_pad, ys_mask, h, hs_mask, return_hidden=True)
        encoding_lengths = mask.sum(-1)[:, -1]
        logits = self.classifier(encodings, encoding_lengths)
        return logits

    def compute_loss(self, logits, labels):
        return self.loss(logits, labels)


class SLU2(SLU):

    def forward(self, feats, feats_lengths):

        src_mask = make_non_pad_mask(feats_lengths.tolist()).to(feats.device).unsqueeze(-2)
        src_mask = src_mask.to(feats.device)
        encodings, mask = self.encoder(feats, src_mask)
        encoding_lengths = mask.sum(-1)[:, -1]
        logits = self.classifier(encodings, encoding_lengths)
        return logits

