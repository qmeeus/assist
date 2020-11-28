import numpy as np

from assist.acquisition import model_factory
from assist.experiment import score, write_scores
from assist.tasks import Structure, coder_factory, read_task, to_string
from assist.tools import FeatLoader, condor_submit, logger, parse_line, read_config


def run_evaluate(expdir, backend="local", cuda=False):
    if backend == "local":
        evaluate(expdir, cuda=cuda)
    elif backend == "condor":
        condor_submit(
            expdir,
            "evaluate",
            [expdir],
            cuda=cuda
        )
    else:
        raise NotImplementedError(f"backend={backend}")


def evaluate(expdir, cuda=False):
    logger.info(f"Evaluate {expdir}")

    acquisitionconf = read_config(expdir/"acquisition.cfg")
    acquisitionconf.set("acquisition", "device", "cuda" if cuda else "cpu")

    coderconf = read_config(expdir/"coder.cfg")
    structure = Structure(expdir/"structure.xml")
    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)

    Model = model_factory(acquisitionconf.get('acquisition', 'name'))
    model = Model(acquisitionconf, coder, expdir)
    model.load(expdir/'model')

    features = FeatLoader(expdir/"testfeats").to_dict()

    with open(expdir/"testtasks") as testtasks:
        references = {
            key: read_task(value)
            for key, value in map(parse_line, testtasks.readlines())
            if key in features
        }

    assert len(features) == len(references)

    to_remove = []
    for uttid, feat in features.items():
        if not np.isfinite(feat).all():
            to_remove.append(uttid)

    if to_remove:
        logger.warning(f"Found {len(to_remove)} utterances with nan.")
        for uttid in to_remove:
            del features[uttid]
            del references[uttid]

    decoded = model.decode(features)

    assert list(decoded) == list(references)
    y_true = np.array([coder.encode(task) for task in references.values()])
    y_pred = np.array([coder.encode(task) for task in decoded.values()])
    from sklearn.metrics import classification_report
    for line in classification_report(y_true, y_pred).split("\n"):
        logger.info(line)

    with open(expdir/"dectasks", "w") as dectasks_file:
        dectasks_file.writelines([
            f"{name}  {to_string(task)}\n"
            for name, task in decoded.items()
        ])

    # TODO: return a dictionary from score instead
    metric_names = ["precision", "recal", "f1", "macro precision", "macro recal", "macro f1"]
    metrics, scores = score(decoded, references)

    for metric_name, metric in zip(metric_names, metrics):
        logger.info(f"{metric_name}: {metric:.4f}")
        with open(expdir/metric_name.replace(" ", ""), "w") as f:
            f.write(str(metric))

    write_scores(scores, expdir)
