import numpy as np
import pickle
import warnings
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from assist.acquisition.classifier import BaseClassifier


warnings.filterwarnings("ignore")


class CustomSVC(SVC):

    def score(self, X, y, sample_weight=None, adjusted=False):
        return balanced_accuracy_score(
            y, self.predict(X),
            sample_weight=sample_weight,
            adjusted=adjusted
        )



class Classifier(BaseClassifier):

    def build(self):
        return

    def _build(self, dummy=False):
        if dummy:
            return DummyClassifier(strategy="uniform")
        return CustomSVC(
            C=float(self.config["C"]),
            kernel=self.config["kernel"],
            tol=float(self.config["tol"]),
            class_weight="balanced",
            probability=True
        )

    def train_loop(self, X, y):
        self.model = [
            self._build(dummy=(len(np.unique(y[:, classid])) == 1))
            .fit(X, y[:, classid])
            for classid in range(self.n_classes)
        ]

    def predict_proba(self, X):
        probabilities = np.zeros((X.shape[0], self.n_classes))
        for classid, clf in enumerate(self.model):
            predictions = clf.predict_proba(X)
            probabilities[:, classid] = (
                predictions[:, 1]               # Case 1: SVM
                if predictions.shape[1] > 1
                else 1 - predictions[:, 0]      # Case 2: DummyClassifier
            )

        return probabilities
