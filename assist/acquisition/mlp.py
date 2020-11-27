import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from assist.acquisition.classifier import BaseClassifier
from assist.tools import logger


class MLP(nn.Module):

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

    def __init__(self, config, coder, expdir):
        super(Classifier, self).__init__(config, coder, expdir)
        self.epochs = int(self.config.get("acquisition", "epochs"))
        self.batch_size = int(self.config.get("acquisition", "batch_size"))
        self.collate_fn = None

    def build(self):
        self.device = torch.device(self.config.get("acquisition", "device"))
        return MLP(
            input_dim=int(self.config.get("acquisition", "input_dim")),
            output_dim=self.n_classes,
            hidden_dim=int(self.config.get("acquisition", "hidden_dim")),
            dropout=float(self.config.get("acquisition", "dropout"))
        ).to(self.device)

    def prepare_inputs(self, features, labels=None):
        tensors = (torch.tensor(features),)
        if labels is not None:
            tensors += (torch.tensor(labels),)
        return TensorDataset(*tensors),

    def train_loop(self, dataset):

        optimizer = torch.optim.Adam(self.model.parameters())

        progress_bar = tqdm(
            total=self.epochs,
            bar_format="{postfix[1][epoch]}/{postfix[0]} loss={postfix[1][loss]:.4f}",
            postfix=[self.epochs, {"epoch": 0, "loss": float('inf')}]
        )

        with progress_bar:
            for epoch in range(self.epochs):
                train_loader = self.get_dataloader(dataset)
                train_loss = self.train_one_epoch(train_loader, optimizer) / len(dataset)
                progress_bar.postfix[1].update({"epoch": epoch, "loss": train_loss})
                progress_bar.update()

    def train_one_epoch(self, dataloader, optimizer):
        epoch_loss = 0
        for *inputs, labels in dataloader:
            optimizer.zero_grad()
            *inputs, labels = self._recursive_to(*inputs, labels)
            loss, _ = self.model(*inputs, labels=labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(dataset, shuffle=is_train, batch_size=self.batch_size)

    def predict_proba(self, dataset):
        with torch.no_grad():
            return torch.sigmoid(torch.cat([
                self.model(*self._recursive_to(*inputs))
                for inputs in self.get_dataloader(dataset, is_train=False)
            ], dim=0)).cpu().numpy()

    def load(self, filename):
        logger.info(f"Loading model from {filename}")
        self.model = torch.load(filename).to(self.device)

    def save(self, filename):
        logger.info(f"Saving model to {filename}")
        torch.save(self.model, filename)

    def _recursive_to(self, *tensors):
        return tuple(tensor.to(self.device) for tensor in tensors)
