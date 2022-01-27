import json
import numpy as np
import pandas as pd
import re
import sys
import torch
import warnings
from collections import namedtuple
from functools import partial
from IPython.display import display
from math import ceil
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("../datasets")
from datasets import Dataset
import assist
from assist.tasks import to_string

USAGE = "Usage: predict.py <OUTDIR> <DATASET_CONFIG> <IKEY> <OKEY> [<TOKENIZER>]"


def mmap(func, *arrays):
    def _func(args):
        return func(*args)
    return map(_func, zip(*arrays))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def onehot(labels, K):
    array = np.zeros((len(labels), K), dtype=int)
    for i, j in enumerate(labels):
        array[i, j] = 1
    return array


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def parse_and_check_args():
    if len(sys.argv) < 5:
        raise TypeError(USAGE)
    outdir, dataset_config, ikey, okey = sys.argv[1:5]
    outdir = Path(outdir)
    assert outdir.is_dir(), f"Not a directory: {outdir}"
    cv, outdirs = (
        (True, list(outdir.glob("cv*")))
        if (outdir/"cv0").exists()
        else (False, [outdir])
    )

    is_nlu = ikey in ("text", "asr")
    if is_nlu and len(sys.argv) < 6:
        raise TypeError(USAGE)
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[5]) if is_nlu else None
    # Options = namedtuple("Options", "outdirs dataset_config ikey okey tokenizer cv".split())
    return outdirs, dataset_config, ikey, okey, tokenizer, cv

def load_data(config, ikey, okey, tokenizer):
    dataset = Dataset(config, ikey, okey, tokenizer=tokenizer)
    return dataset


def predict(outdir, dataset, device="cpu"):
    if dataset.has_subset("test"):
        test_set = dataset("test")
    else:
        test_set = dataset(indices=f"{outdir}/test.txt")

    batch_size = 40
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_set.data_collator,
        drop_last=False
    )

    model = torch.load(f"{outdir}/ckpt-best.pt", map_location=device)

    y_pred = []
    for inputs, input_lengths, _ in tqdm(loader, total=ceil(len(test_set)/batch_size)):
        with torch.no_grad():
            inputs, input_lengths = (t.to(device) for t in (inputs, input_lengths))
            logits = model(inputs, input_lengths).detach().cpu()
        y_pred.extend(logits.tolist())
    return y_pred, test_set


def evaluate_multiclass(dataset, y_pred, test_set, outdir, speakers=False):
    y_pred = np.array(y_pred)
    predictions = y_pred.argmax(-1)
    labels = test_set.labels[:len(predictions)]
    index = test_set.indices[:len(predictions)]

    names = np.array(dataset.config["classes"])[sorted(set(labels).union(set(predictions)))]

    with open(f"{outdir}/classification_report.txt", "w") as f:
        f.write(classification_report(labels, predictions, target_names=names))

    pd.DataFrame(
        confusion_matrix(labels, predictions),
        index=names, columns=names
    ).to_csv(f"{outdir}/confusion_matrix.csv")

    K = len(dataset.config["classes"])
    results = {"accuracy": accuracy_score(labels, predictions)}
    for method in ("macro", "micro", "weighted", None):
        key = method or "classes"
        results[key] = dict(zip(
            "precision recall fscore support".split(),
            precision_recall_fscore_support(labels, predictions, average=method)
        ))
        ohlabels, ohpredictions = (onehot(array, K) for array in (labels, predictions))
        mask = (ohlabels.sum(0) != 0) & (ohpredictions.sum(0) != 0)
        results[key]["roc_auc"] = roc_auc_score(
            ohlabels[:, mask], ohpredictions[:, mask],
            average=method, multi_class="ovo"
        )

    if speakers:
        speakers = np.array(list(map(lambda s: re.sub(r"_\w+$", "", s), index)))
        results["speakers"] = {
            spkr: accuracy_score(labels[speakers == spkr], predictions[speakers == spkr])
            for spkr in speakers
        }


    with open(f"{outdir}/results.json", "w") as f:
        json.dump(results, f, indent=4, default=default)

    print(f"Accuracy: {results['accuracy']:.3%}")
    display(pd.DataFrame.from_dict(results).drop(["accuracy", "classes"], axis=1).drop("support"))
    return results


def evaluate_multilabel(dataset, y_pred, test_set, outdir, speakers=False):
    decode = partial(dataset.coder.decode, cost=log_loss)
    predictions = list(map(to_string, map(decode, map(np.array, y_pred))))
    with open(f"{outdir}/predictions.txt", "w") as f:
        f.writelines(mmap("{} {}\n".format, test_set.indices, predictions))

    labels = list(map(to_string, map(decode, test_set.labels)))
    with open(f"{outdir}/labels.txt", "w") as f:
        f.writelines(mmap("{} {}\n".format, test_set.indices, labels))

    ps = (sigmoid(np.array(y_pred)) > .5).astype(int)
    error_before = 1 - (ps == test_set.labels).all(-1).mean()
    error_after = 1 - (np.array(predictions) == np.array(labels)).mean()
    print("Error rate: ")
    print(f"  before decoding: {error_before:.4f}")
    print(f"  after decoding: {error_after:.4f}")

    with open(f"{outdir}/results.json", "r") as f:
        results = json.load(f)
    results["test_error_before"] = error_before
    results["test_error_after"] = error_after
    with open(f"{outdir}/results.json", "w") as f:
        json.dump(results, f, indent=4, default=default)

    return results


def main():

    outdirs, dataset_config, ikey, okey, tokenizer, cv = parse_and_check_args()
    dataset = load_data(dataset_config, ikey, okey, tokenizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate = evaluate_multilabel if okey == "tasks" else evaluate_multiclass

    results = []
    for outdir in outdirs:
        y_pred, test_set = predict(outdir, dataset, device=device)
        results.append(evaluate(dataset, y_pred, test_set, outdir, speakers=True))

    if cv:
        averages = {k: np.mean([d[k] for d in results]) for k in results[0] if k != "classes"}
        stdevs = {k: np.std([d[k] for d in results]) for k in results[0] if k != "classes"}
        with open(outdirs[0].parent/"results.json", "w") as f:
            summary = {"cv_means": averages, "cv_stdevs": stdevs, "cv_results": results}
            json.dump(summary, f, indent=4, default=default)



if __name__ == "__main__":
    main()

