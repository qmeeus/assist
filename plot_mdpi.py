from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import seaborn as sns

def split_path2(path):
    feats, runid, _, spkr, expid, fold = path.parts[-7:-1]
    numblocks, trainsize = expid.split("_")
    numblocks = int(numblocks.replace("block", ""))
    return feats, runid, spkr, numblocks, int(trainsize), int(fold.replace("cv", ""))


def load_cv(outdir):
    index_names = ["feats","runid", "speaker", "numblocks", "trainsize", "fold"]
    result_files = Path(outdir).rglob("results.json")
    results = pd.concat({
        split_path2(path): pd.read_json(path, typ="series").rename("value").rename_axis("metric")
        for path in result_files
    }, axis=0, names=index_names)
    return results.unstack(-1).reset_index(drop=False)



def split_path(path):
    model, spkr, exp, metric = path.parts[-4:]
    numblocks, expid = exp.split("_")
    numblocks = int(numblocks.replace("blocks", ""))
    expid = int(expid.replace("exp", ""))
    with open(path.parent/"traintasks") as f:
        numtrain = len(f.readlines())
    return model, spkr, numblocks, numtrain, expid

def read_score(filename):
    with open(filename, "r") as f:
        return float(f.read().strip())


def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    df = (
        pd.concat({key: pd.DataFrame(value) for key, value in data.items()})["val_acc"]
        .sort_values().rename("accuracy").groupby(level=0).tail(1).to_frame().droplevel(1).reset_index(drop=False)
        .assign(expid=lambda df: df["index"].map(lambda s: int(s.replace("cv", "")))).drop(columns="index")
    )

    speaker, expid = filename.parts[-3:-1]
    expid, trainsize = expid.split("_")
    expid = int(expid.replace("block", ""))
    trainsize = int(trainsize)
    df["model"], df["speaker"], df["trainsize"] = "clstransformer", speaker, trainsize
    df.set_index(["model", "speaker", "trainsize", "expid"], inplace=True)
    return df


def load_cv(outdir):
    index_names = ["feats","runid", "speaker", "numblocks", "trainsize", "fold"]
    result_files = Path(outdir).rglob("results.json")
    results = pd.concat({
        split_path2(path): pd.read_json(path, typ="series").rename("value").rename_axis("metric")
        for path in result_files
    }, axis=0, names=index_names)
    return results.unstack(-1).reset_index(drop=False)


def plot_patience():
    nmf_dir = Path("/esat/spchdisk/scratch/hvanhamm/assist/expdir/patience/char/ctc/NMF_1_2_ctc_char")
    nmf_results = (
        pd.DataFrame({
            split_path(path): [read_score(path)]
            for d in map(Path, [nmf_dir]) for path in d.rglob("f1")
        }).T
        .rename(columns={0: "micro F1"})
        .rename_axis(["feats", "speaker", "numblocks", "trainsize", "fold"])
        .reset_index(drop=False)
    )
    import ipdb; ipdb.set_trace()
    results = pd.concat(list(map(load_cv, [
        "exp/patience/clm",
        "exp/patience/mlm",
        "exp/patience/text",
        "exp/patience/pipeline"
    ])), axis=0).reset_index(drop=True)
    results.loc[results.feats == "mlm", "runid"].unique()
    results.loc[results["runid"] == "20220413_171002", "feats"] = "mlm2"
    results = results.where(results.runid != "20220413_171002").dropna(how="all")
    plot_data = (
        pd.concat([results, nmf_results], axis=0)
        .dropna(how='any', axis=1)
        .reset_index(drop=True)
        .assign(**{
            "feats": lambda df: df.feats.replace("NMF_1_2_ctc_char", "NMF").map(lambda s: s if len(s) > 3 else s.upper()),
            "train size": lambda df: pd.qcut(df.trainsize, np.linspace(0, 1, 11)).map(lambda x: x.mid)
        })
        .sort_values("feats")
    )
    sns.lineplot(data=plot_data, x="train size", y="micro F1", hue="feats")
    plt.tight_layout()
    plt.savefig("tests/outputs/compare-results-patience.pdf", dpi=300)



def plot_grabo():
    assist_root = "/esat/spchtemp/scratch/qmeeus/repos/assist"
    assist_dirs = list(map(lambda s: f"{assist_root}/{s}", [
        "exp/grabo/gru_128_mlm_wwm_batch_norm",
        "exp/grabo/lstm_128_asr",
        "exp/grabo/lstm_128_text",
        "exp/grabo/lstm_clm_2800it"
    ])) + ["/esat/spchdisk/scratch/hvanhamm/assist/expdir/grabo/char/ctc/NMF_1_2_ctc_char"]
    assist_results = (
        pd.DataFrame({
            split_path(path): [read_score(path)]
            for d in map(Path, assist_dirs) for path in d.rglob("f1")
        }).T
        .rename(columns={0: "micro F1"})
        .rename_axis(["feats", "speaker", "numblocks", "trainsize", "fold"])
        .reset_index(drop=False)
    )
    assist_results["feats"] = assist_results["feats"].replace(dict(zip(assist_results.feats.unique(), "MLM pipeline text CLM NMF".split())))
    assist_results.sort_values("feats", inplace=True)
    grabo_results = pd.concat([load_cv("exp/grabo/mlm"), assist_results], axis=0).dropna(how="any", axis=1).reset_index(drop=True)
    grabo_results["train size"] = pd.qcut(grabo_results["trainsize"], np.linspace(0, 1, 11)).map(lambda x: x.mid)
    sns.lineplot(data=grabo_results, x="train size", y="micro F1", hue="feats")
    plt.tight_layout()
    plt.savefig("tests/outputs/grabo-compare-all.pdf", dpi=300)

def main():
    # plot_patience()
    plot_grabo()


if __name__ == "__main__":
    main()
