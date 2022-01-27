import numpy as np

from sklearn.metrics import classification_report

from assist.acquisition import model_factory
from assist.experiment import score, write_scores
from assist.tasks import Structure, coder_factory, read_task, to_string
from assist.tools import FeatLoader, condor_submit, logger, parse_line, read_config


def run_evaluate(expdir, backend="local", cuda=False, clean=False):
    if backend == "local":
        evaluate(expdir, cuda=cuda, clean=clean)
    elif backend == "condor":
        condor_submit(
            expdir,
            "evaluate",
            [expdir],
            command_options="--clean" if clean else "",
            cuda=cuda
        )
    else:
        raise NotImplementedError(f"backend={backend}")


def evaluate(expdir, cuda=False, clean=False):
    logger.info(f"Evaluate {expdir}")

    acquisitionconf = read_config(expdir/"acquisition.cfg")
    acquisitionconf.set("acquisition", "device", "cuda" if cuda else "cpu")

    coderconf = read_config(expdir/"coder.cfg")
    structure = Structure(expdir/"structure.xml")
    Coder = coder_factory(coderconf.get('coder', 'name'))
    coder = Coder(structure, coderconf)

    Model = model_factory(acquisitionconf.get('acquisition', 'name'))
    model = Model(acquisitionconf, coder, expdir)
    model.display(logger.info)
    model.load(expdir/'model')
    model.display(logger.info)

    features = FeatLoader(expdir/"testfeats").to_dict()

    with open(expdir/"testtasks") as testtasks:
        references = {
            key: read_task(value)
            for key, value in map(parse_line, testtasks.readlines())
            if key in features
        }

    assert len(features) == len(references), set(features) - set(references)

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

    assert not(set(decoded) - set(references))
    y_true = np.array([coder.encode(task) for task in references.values()])
    y_pred = np.array([coder.encode(task) for task in decoded.values()])

    TP = (y_pred == 1) & (y_true == 1)
    TN = (y_pred == 0) & (y_true == 0)
    FP = (y_pred == 1) & (y_true == 0)
    FN = (y_pred == 0) & (y_true == 1)
    error_rate = 1 - (TP | TN).all(-1).mean()
    precision = TP.sum() / (TP | FP).sum()
    recall = TP.sum() / (TP | FN).sum()
    f1_score = 2 * precision * recall / (precision + recall)
    logger.info(f"TPR={TP.sum()} TN={TN.sum()} FP={FP.sum()} FN={FN.sum()}")
    logger.info(f"P={precision:.2%} R={recall:.2%} F={f1_score:.2%} E={error_rate:.2%}")


    for line in classification_report(y_true, y_pred).split("\n"):
        logger.info(line)

    with open(expdir/"dectasks", "w") as dectasks_file:
        dectasks_file.writelines([
            f"{name} {to_string(task)}\n"
            for name, task in decoded.items()
        ])

    metrics, scores = score(decoded, references)
    for metric_name, metric in metrics.items():
        logger.info(f"{metric_name}: {metric:.4f}")
        with open(expdir/metric_name.replace(" ", ""), "w") as f:
            f.write(str(metric))

    write_scores(scores, expdir)

    if clean:
        logger.info(f"Remove {expdir}/model")
        os.remove(expdir/'model')
