import numpy as np
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
        self.epochs = int(self.config["epochs"])
        self.batch_size = int(self.config["batch_size"])
        self.iterations = int(self.config.get("iterations", 0))
        self.collate_fn = None

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

    def train_loop(self, dataset):
        if self.iterations > 0:
            self.max_iter = self.iterations
            self.epochs = 0
        else:
            self.max_iter = int(np.ceil(self.epochs * len(dataset) / self.batch_size))
        assert any([self.epochs, self.iterations])
        logger.debug(f"Number of iterations: {self.iterations}")

        optimizer = torch.optim.Adam(self.model.parameters())

        progress_bar = tqdm(
            total=self.max_iter,
            bar_format="{postfix[1][iter]}/{postfix[0]} loss={postfix[1][loss]:.4f}",
            postfix=[self.max_iter, {"iter": 0, "loss": float('inf')}]
        )

        with progress_bar:
            iteration = 0
            while True:
                train_iter = iter(self.batch_iterator(dataset, is_train=True))
                iteration += 1
                *inputs, labels = next(train_iter)
                train_loss = self.train_one_step(inputs, labels, optimizer) / len(labels)
                progress_bar.postfix[1].update({"iter": iteration, "loss": train_loss})
                progress_bar.update()
                if iteration == self.max_iter:
                    break

    def train_one_step(self, inputs, labels, optimizer):
        optimizer.zero_grad()
        *inputs, labels = self._recursive_to(*inputs, labels)
        loss, _ = self.model(*inputs, labels=labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def batch_iterator(self, dataset, is_train=True):
        if is_train:
            epochs = self.epochs if self.epochs > 0 else float("inf")
        else:
            epochs = 1
        epoch = 0
        while epoch < epochs:
            yield from iter(self.get_dataloader(dataset, is_train))
            epoch += 1

    def get_dataloader(self, dataset, is_train=True):
        return DataLoader(dataset, shuffle=is_train, batch_size=self.batch_size)

    def predict_proba(self, dataset):
        with torch.no_grad():
            return torch.sigmoid(torch.cat([
                self.model(*self._recursive_to(*inputs))
                for inputs in self.batch_iterator(dataset, is_train=False)
            ], dim=0)).cpu().numpy()

    def load(self, filename):
        logger.info(f"Loading model from {filename}")
        self.model = torch.load(str(filename), map_location=self.device)

    def save(self, filename):
        logger.info(f"Saving model to {filename}")
        torch.save(self.model, filename)

    def _recursive_to(self, *tensors):
        return tuple(tensor.to(self.device) for tensor in tensors)

    @staticmethod
    def display_model(model):
        for line in str(model).split("\n"):
            logger.info(line)
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
