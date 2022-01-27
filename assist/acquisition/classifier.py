import numpy as np
import pickle
import torch
import warnings
from configparser import ConfigParser
from functools import partial
from sklearn.metrics import log_loss

from assist.tasks import Task, read_task
from assist.tools import logger

warnings.filterwarnings("ignore")


class BaseClassifier:

    def __init__(
        self,
        config=None,
        coder=None,
        expdir=None
    ):

        if isinstance(config, ConfigParser):
            self.config = dict(config["acquisition"].items())
        else:
            self.config = config

        self.coder = coder
        self.expdir = expdir
        self.n_classes = self.config.get("output_dim", None) or coder.numlabels  # Backward compat
        self.model = self.build()
        logger.debug(f"Num classes: {self.n_classes}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expdir})"

    def build(self):
        raise NotImplementedError

    def train_loop(self, *inputs):
        raise NotImplementedError

    def predict_proba(self, *inputs):
        raise NotImplementedError

    def encode(self, tasks):
        if isinstance(tasks, str):
            return self.coder.encode(read_task(tasks))
        if isinstance(tasks, Task):
            return self.coder.encode(tasks)
        if isinstance(tasks, list):
            return np.array([self.encode(task) for task in tasks])
        raise TypeError(f"{type(tasks)}")

    def encode_target(self, taskstring):
        """
        Encode the string representation of a task into a 1D vector
        Parameters
        ----------
        taskstring : str
            XML representation of a task, read by read_task and passed to coder.encode
        Returns : 1D array
            numpy array of dimension [n_classes,]
        """
        # if isinstance(taskstring, list):
        #     return list(map(self.encode_target, taskstring))
        # return self.coder.encode(read_task(taskstring))
        return self.encode(taskstring)

    def train(self, examples, test_examples=None):
        logger.debug(f"{len(examples)} training examples")
        features, tasks = zip(*examples.values())
        target = [self.encode_target(task) for task in tasks]
        if test_examples is not None:
            test_feats, test_tasks = zip(*examples.values())
            test_target = [self.encode_target(task) for task in test_tasks]
            self.fit(features, target, (test_feats, test_target))
            return
        self.fit(features, target)

    def fit(self, X, y, test_set=None):
        inputs = self.prepare_inputs(X, y)
        # if test_set is not None:
        #     inputs += self.prepare_inputs(*test_set)

        self.train_loop(*inputs)
        return self

    def prepare_inputs(self, features, labels=None):
        """
        Prepare the inputs for training
        Parameters
        ----------
        features : Array-like
            Contains the features for training
        labels : Array-like
            If given, contains the target
        returns : Tuple[Any]
            By default, numpy arrays, but whatever is required by `train_loop`
        """
        inputs = (np.stack(features, axis=0),)
        if labels is not None:
            inputs += (np.array(labels, dtype=int),)
        return inputs

    def _decode(self, dataset):
        _type = type(dataset)
        inputs = self.prepare_inputs(dataset)
        probs = self.predict_proba(*inputs)
        cost_func = partial(log_loss, normalize=False)
        # TODO: parallelize (jobs > 1) or use Cython
        return _type(list(map(partial(self.coder.decode, cost=cost_func), probs)))

    def decode(self, examples):
        names = list(examples.keys())
        predictions = self._decode(list(examples.values()))
        # return dict(zip(names, map(partial(self.coder.decode, cost=cost_func), probs)))
        return dict(zip(names, predictions))

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
