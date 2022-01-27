import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from assist.acquisition.classifier import BaseClassifier as _BaseClassifier
from .utils import display_model
from assist.tools import logger, load_json


class BaseClassifier(_BaseClassifier):
    """
    Compat class pytorch model -> assist compatible model
    """

    def __init__(self, config, coder, expdir):
        super(BaseClassifier, self).__init__(config, coder, expdir)
        self.epochs = int(self.config["epochs"])
        self.batch_size = int(self.config["batch_size"])
        self.learning_rate = float(self.config.get("lr", 1e-2))
        self.iterations = int(self.config.get("iterations", 0))
        self.collate_fn = None

    def build(self):
        raise NotImplementedError

    def prepare_inputs(self, features, labels=None):
        raise NotImplementedError

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_learning_rate_scheduler(self, optimizer):
        lr_scheduler_type = self.config.get("lr_scheduler", None)
        if not lr_scheduler_type:
            return
        LRScheduler = getattr(torch.optim.lr_scheduler, lr_scheduler_type)
        cfg = self.config.get("lr_scheduler_config", None)
        sched_cfg = load_json(cfg) if cfg is not None else {}
        return LRScheduler(optimizer, **sched_cfg)

    def set_batch_size(self, train_size):
        if train_size < self.batch_size:
            logger.info(f"Setting batch size to {train_size}")
            self.batch_size = train_size
        # batch_sizes = {50: 8, 100: 16, 200: 32}
        # for sz, bs in batch_sizes.items():
        #     if train_size <= sz:
        #         self.batch_size = bs
        #         return

    def train_loop(self, dataset):
        self.model.train()
        self.set_batch_size(len(dataset))
        self.max_iter = self.get_num_iter(len(dataset), self.batch_size, self.iterations, self.epochs)
        epochs = self.iter_to_epochs(self.max_iter, len(dataset), self.batch_size)
        logger.info(f"{self.max_iter} iterations ~ {epochs:.2f} epochs / ({len(dataset)} examples / {self.batch_size} examples per batch)")

        optimizer = self.get_optimizer()
        lr_scheduler = self.get_learning_rate_scheduler(optimizer)

        progress_bar = tqdm(
            total=self.max_iter,
            bar_format="num_examples={postfix[2]} {postfix[1][iter]}/{postfix[0]} loss={postfix[1][loss]:.4f}",
            postfix=[self.max_iter, {"iter": 0, "loss": float('inf')}, len(dataset)]
        )

        with progress_bar:
            iteration = 0
            while True:
                train_iter = iter(self.batch_iterator(dataset, is_train=True))
                iteration += 1
                *inputs, labels = next(train_iter)
                train_loss = self.train_one_step(inputs, labels, optimizer, lr_scheduler) / len(labels)
                progress_bar.postfix[1].update({"iter": iteration, "loss": train_loss})
                progress_bar.update()
                if iteration == self.max_iter:
                    break

    def train_one_step(self, inputs, labels, optimizer, lr_scheduler=None):
        if lr_scheduler is not None:
            lr_scheduler.step()
            logger.debug(f"LR: {lr_scheduler.get_lr()}")
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
        raise NotImplementedError

    def predict_proba(self, dataset):
        self.model.eval()
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

    def display(self, print_fn=logger.info):
        display_model(self.model, print_fn)

    @staticmethod
    def get_num_iter(dataset_size, batch_size, iterations=None, epochs=None):
        """
        iterations is the maximum number of batches
        epochs is the minimum number of times the same example will be observed during training
        """
        # if iterations:
        #     epochs = BaseClassifier.iter_to_epochs(iterations, dataset_size, batch_size)
        # return BaseClassifier.epochs_to_iter(
        #     epochs * 3 if dataset_size < 200 else epochs, dataset_size, batch_size)

        assert any(n>0 for n in (iterations, epochs))
        if not(iterations):
            return BaseClassifier.epochs_to_iter(epochs, dataset_size, batch_size)
        elif not(epochs):
            return iterations
        else:
            epochs_as_iter = BaseClassifier.epochs_to_iter(epochs, dataset_size, batch_size)
            return max(iterations, epochs_as_iter)

    @staticmethod
    def epochs_to_iter(epochs, dataset_size, batch_size):
        return int(np.ceil(epochs * (dataset_size / batch_size)))

    @staticmethod
    def iter_to_epochs(n_iter, dataset_size, batch_size):
        return n_iter / (dataset_size / batch_size)