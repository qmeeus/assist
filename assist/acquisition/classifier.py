import numpy as np
import pickle
import warnings
from configparser import ConfigParser
from functools import partial
from sklearn.metrics import log_loss

from assist.tasks import read_task
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
        return self.coder.encode(read_task(taskstring))

    def train(self, examples):
        logger.debug(f"{len(examples)} training examples")
        features, tasks = zip(*examples.values())
        target = [self.encode_target(task) for task in tasks]
        self.fit(features, target)

    def fit(self, X, y):
        inputs = self.prepare_inputs(X, y)
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

    def decode(self, examples):
        names = list(examples.keys())
        inputs = self.prepare_inputs(list(examples.values()))
        probs = self.predict_proba(*inputs)
        cost_func = partial(log_loss, normalize=False)
        return dict(zip(names, map(partial(self.coder.decode, cost=cost_func), probs)))

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
