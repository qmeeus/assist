import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from assist.acquisition.lstm import Classifier


def error_rate_scorer(estimator, X, y):
    return (estimator.predict(X) == y).all(axis=1).mean()


class RNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        input_dim=768, 
        output_dim=32, 
        hidden_dim=128,
        num_layers=1,
        dropout=.1, 
        rnn_type="lstm", 
        device="cuda", 
        epochs=20,
        iterations=0,
        batch_size=128,
    ):

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.rnn_type = rnn_type
        self.device = device
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.iterations = iterations
        
    def fit(self, X, y):
        
        # FIXME: horrible hack
        config = {
            "input_dim": self.input_dim, 
            "output_dim": self.output_dim, 
            "hidden_dim": self.hidden_dim, 
            "num_layers": self.num_layers,
            "dropout": self.dropout, 
            "rnn_type": self.rnn_type,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "iterations": self.iterations
        }

        self.model_ = Classifier(config, None, None)
        inputs = self.model_.prepare_inputs(X, y)
        self.model_.train_loop(*inputs)
        return self

    def predict(self, X):

        check_is_fitted(self)
        inputs = self.model_.prepare_inputs(X)
        probs = self.model_.predict_proba(*inputs)
        return (probs > .5).astype(int)
