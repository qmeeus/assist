import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from assist.acquisition.torch_models.base import BaseClassifier


class MLP(nn.Module):
    """
    Multilayer perceptron
    TODO: add sequence -> vector operation here (select first, average, encoder, etc.)
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, features, labels=None):
        logits = self.layers(features)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits
        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fct(logits, labels.float())


class Classifier(BaseClassifier):

    def build(self):
        self.device = torch.device(self.config["device"])
        model = MLP(
            input_dim=int(self.config["input_dim"]),
            output_dim=self.n_classes,
            hidden_dim=int(self.config["hidden_dim"]),
            dropout=float(self.config["dropout"])
        ).to(self.device)
        self.display_model(model)
        return model

    def prepare_inputs(self, features, labels=None):
        tensors = (torch.tensor(features),)
        if labels is not None:
            tensors += (torch.tensor(labels),)
        return TensorDataset(*tensors),

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(dataset, shuffle=is_train, batch_size=self.batch_size)
