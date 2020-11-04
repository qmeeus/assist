import numpy as np
import pickle
from functools import partial
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from assist.tasks.read_task import read_task


class CustomSVC(SVC):

    def score(self, X, y, sample_weight=None, adjusted=False):
        return balanced_accuracy_score(
            y, self.predict(X),
            sample_weight=sample_weight,
            adjusted=adjusted
        )



class Classifier:

    def __init__(self, config, coder, expdir):
        self.config = config
        self.coder = coder
        self.expdir = expdir
        self.n_classes = coder.numlabels
        self.classifiers = None

    def build(self, dummy=False):
        if dummy:
            return DummyClassifier("most_frequent")
        return CustomSVC(
            C=float(self.config.get("acquisition", "C")),
            kernel=self.config.get("acquisition", "kernel"),
            tol=float(self.config.get("acquisition", "tol")),
            class_weight="balanced",
            probability=True
        )

    def train(self, examples):
        features, tasks = zip(*examples.values())
        features = np.stack(features, axis=0)
        tasks = [read_task(task) for task in tasks]
        noisetype = self.coder.conf['noisetype']
        noiseprob = float(self.coder.conf['noiseprob'])
        target = np.array([
            self.coder.encode(t,noisetype,noiseprob) for t in tasks
        ], dtype=int)

        assert target.shape[1] == self.n_classes

        self.classifiers = [
            self.build(dummy=(len(np.unique(target[:, classid])) == 1))
            .fit(features, target[:, classid])
            for classid in range(self.n_classes)
        ]

    def decode(self, examples):
        assert self.classifiers
        names = list(examples.keys())
        X = np.stack(list(examples.values()), axis=0)
        probabilities = np.zeros((X.shape[0], self.n_classes))
        for classid, clf in enumerate(self.classifiers):
            predictions = clf.predict_proba(X)
            probabilities[:, classid] = (
                predictions[:, 1]
                if predictions.shape[1] > 1
                else 1 - predictions[:, 0]
            )

        cost_func = partial(log_loss, normalize=False)
        return dict(zip(names, map(
            partial(self.coder.decode, cost=cost_func),
            probabilities
        )))

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.classifiers, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.classifiers = pickle.load(f)
