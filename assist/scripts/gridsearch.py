import json
import numpy as np
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from itertools import product
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from time import time

from assist.acquisition import model_factory
from assist.acquisition.sklearn_model import RNNClassifier
from assist.tasks import Structure, coder_factory, read_task
from assist.tools import FeatLoader, logger, parse_line, read_config
from assist.scripts.train import prepare_subset


def prepare_gridsearch(expdir, recipe):

    os.makedirs(expdir, exist_ok=True)
    for filename in ["param_grid.json", "gridsearch.cfg", "coder.cfg", "structure.xml"]:
        logger.debug(f"Copy {filename} from {recipe} to {expdir}")
        shutil.copy(recipe/filename, expdir/filename)

    dataconf = read_config(recipe/"database.cfg")
    prepare_subset(expdir, "gridsearch", dataconf)


# def gridsearch(expdir, recipe, cuda=True, n_jobs=1):
#     logger.info(f"GridSearch {expdir}")

#     with open(recipe/"param_grid.json") as jsonfile:
#         param_grid = json.load(jsonfile)
        
#     logger.debug(str(param_grid))
#     total_params = np.prod(list(map(len, param_grid.values())))
#     logger.info(f"Searching {len(param_grid)} parameters, totalling {total_params} possible values.")

#     model_config = dict(read_config(expdir/"gridsearch.cfg")["acquisition"].items())
#     model_config["device"] = "cuda" if cuda else "cpu"

#     coderconf = read_config(expdir/"coder.cfg")
#     structure = Structure(expdir/'structure.xml')
#     Coder = coder_factory(coderconf.get('coder', 'name'))
#     coder = Coder(structure, coderconf)
#     model_config["output_dim"] = coder.numlabels

#     Model = model_factory(acquisitionconf.get('acquisition', 'name'))
#     model = RNNClassifier(Model, model_config, coder, expdir)
#     gs = GridSearchCV(model, param_grid, n_jobs=n_jobs, refit=False)
    
#     features = FeatLoader(expdir/"gridsearchfeats").to_dict()
#     with open(expdir/"gridsearchtasks") as traintasks:
#         taskstrings = {
#             uttid: task
#             for uttid, task in map(parse_line, traintasks.readlines())
#         }

#     indices = sorted(set(features).intersection(set(taskstrings)))
#     X = list(map(features.__getitem__, indices))
#     y = list(map(coder.encode, map(read_task, map(taskstrings.__getitem__, indices))))

#     gs.fit(X, y)
#     with open(expdir/"gs_results.json", "w") as result_file:
#         json.dump({
#             "best_params": gs.best_params_, 
#             "best_score": gs.best_score_, 
#             "cv_results": gs.cv_results_
#         }, result_file, indent=4)

    
def gs_learning_curve(expdir, recipe, cuda=True, n_jobs=1):
    logger.info(f"GridSearch {expdir}")

    with open(recipe/"param_grid.json") as jsonfile:
        param_grid = json.load(jsonfile)
        
    logger.debug(str(param_grid))
    total_params = np.prod(list(map(len, param_grid.values())))
    logger.warning(f"Searching {len(param_grid)} parameters, totalling {total_params} possible values.")

    gsconf = read_config(expdir/"gridsearch.cfg")
    default_config = dict(gsconf["acquisition"].items())
    default_config["device"] = "cuda" if cuda else "cpu"
    gsconf = dict(gsconf["gridsearch"].items())
    logger.debug(" ".join(f"{k}={v}" for k, v in gsconf.items()))
    train_sizes = np.linspace(float(gsconf["nmin"]), float(gsconf["nmax"]), int(gsconf["num_trains"]))
    gs_params = {
        "train_sizes": train_sizes,
        "cv": int(gsconf["cv_splits"]),
        "scoring": make_scorer(accuracy) if gsconf["scoring"] == "accuracy" else gsconf["scoring"],
        "n_jobs": n_jobs
    }
    logger.debug(gs_params)

    coderconf = read_config(expdir/"coder.cfg")
    structure = Structure(expdir/'structure.xml')
    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)
    default_config["output_dim"] = coder.numlabels

    features = FeatLoader(expdir/"gridsearchfeats").to_dict()
    with open(expdir/"gridsearchtasks") as traintasks:
        taskstrings = {
            uttid: task
            for uttid, task in map(parse_line, traintasks.readlines())
        }

    indices = sorted(set(features).intersection(set(taskstrings)))
    X = list(map(features.__getitem__, indices))
    y = list(map(coder.encode, map(read_task, map(taskstrings.__getitem__, indices))))

    gs_results = defaultdict(list)
    start = time()
    best_score = 0
    for i, param_values in enumerate(product(*param_grid.values())):

        t0 = time()
        params = dict(zip(param_grid.keys(), param_values))
        config = deepcopy(default_config)
        config.update(params)
        logger.debug(config)

        model = RNNClassifier(**config)

        train_sizes, train_scores, valid_scores = learning_curve(model, X, y, **gs_params)

        train_score = auc(train_sizes, train_scores.mean(-1))
        test_score = auc(train_sizes, valid_scores.mean(-1))
        t1 = time()
        logger.info(
            f"model {i+1}/{total_params}: train={train_score:.3%} test={test_score:.3%} "
            f"time={t1 - t0:.1f}s elapsed={t1-start:.1f}s {params}"
        )
        gs_results["auc_test_score"].append(test_score)
        gs_results["auc_train_score"].append(train_score)
        gs_results["params"].append(params)
        gs_results["train_sizes"].append(train_sizes)
        gs_results["train_scores"].append(train_scores)
        gs_results["test_scores"].append(valid_scores)

        if test_score > best_score:
            best_params, best_score, best_index = params, test_score, i

    logger.warning(
        f"Search completed in {time() - start:.2f}s. Best model: {best_params} ({best_score:.2%})"
    )
    logger.warning(f"Test scores: {gs_results['test_scores'][best_index].mean(-1)}")

    with open(expdir/"gs_results.json", "w") as result_file:
        json.dump({"best_params": best_params, "best_score": best_score, "cv_results": serialise(gs_results)}, result_file)


def serialise(obj):
    if isinstance(obj, (int, str, float)):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return dict(map(serialise, obj.items()))
    elif iter(obj):
        return list(map(serialise, obj))
    else:
        raise TypeError(f"Unexpected type: {type(obj)}")


def accuracy(y_true, y_pred):
    return (y_true == y_pred).all(-1).mean()


def auc(train_sizes, valid_scores):
    x = (train_sizes - train_sizes.min()) / (train_sizes.max() - train_sizes.min())
    dx, dy = (np.diff(array) for array in (x, valid_scores))
    y0 = valid_scores[:-1]
    return (dx * y0 + .5 * dx * dy).sum()


