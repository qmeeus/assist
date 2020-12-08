import h5py
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .tools import isin, parse_line
from .logger import logger


__all__ = [
    "FeatLoader",
    "load_hdf5",
    "load_numpy",
    "save_features",
    "save_hdf5",
    "save_numpy",
    "save_scp",
]


DataDict = Dict[str,np.ndarray]


def save_features(
        features:DataDict,
        filename:Path,
        storage:Optional[str]="numpy",
        feature_name:Optional[str]="features"
) -> None:
    """
    Helper function to quickly save features to file and write scp registry
    Parameters
    ----------
    features : Dict[str,ndarray]
        Data to be stored on disk
    filename: Pathlike
        Name of the directory where to save the data
    storage : str (optional)
        hdf5 for hdf5 storage, else numpy
    Returns : None
    """
    storage = "hdf5" if storage == "hdf5" else "numpy"
    save_func = save_hdf5 if storage == "hdf5" else save_numpy
    filename = save_func(features, filename)
    logger.info(f"Features saved to {filename}")
    save_scp(filename, list(features), filename.parent/f"{filename.stem}.scp")


def save_scp(target_file:Path, identifiers:List[str], filename:Path) -> None:
    """
    Save a registry with key-locations in a kaldi-style SCP-like fashion
    Parameters
    ----------
    target_file : Pathlike
        The name of the file where the data is stored
    identifiers : List[str]
        The keys that identify the data in `target_file`
    filename : Pathlike
        The name of the output file
    Returns : None
    """
    with open(filename, "w") as scpfile:
        scpfile.writelines([
            f"{key} {target_file}:{key}\n" for key in identifiers
        ])


def save_hdf5(arrays:DataDict, filename:Path) -> None:
    """
    Save a dictionary of arrays of different lengths to an h5 database
    Parameters
    ----------
    arrays : Dict[str,ndarray]
        A dictionary where the keys are the identifiers for storing the arrays in the h5 database
    filename : Pathlike
        The name of the output file (without extension, the extension h5 will be added)
    Returns : filename
    """
    compression = {"compression": "gzip", "compression_opts": 9}
    filename = filename.parent/(filename.name + ("" if filename.suffix == ".h5" else ".h5"))
    with h5py.File(filename, "w") as h5f:
        for key, array in arrays.items():
            h5f.create_dataset(key, data=array, **compression_opts)
    return filename


def save_numpy(arrays:DataDict, filename:Path) -> None:
    """
    Save a dictionary of arrays to numpy npz format
    Parameters
    ----------
    arrays : Dict[str,ndarray]
        A dictionary where the keys are the identifiers for storing the arrays
    filename : Pathlike
        The name of the output file
    Returns : filename
    """
    filename = filename.parent/(filename.name + ("" if filename.suffix == ".npz" else ".npz"))
    np.savez_compressed(filename, **arrays)
    return filename


def load_hdf5(filename:Path, filter_keys:Optional[Union[List[str],Callable]]=None) -> DataDict:
    """
    Load the contents of an hdf5 database as a dictionary where keys are the identifiers in the
    database and values are the contents. Takes optionally a `filter_keys` argument to control
    which keys are loaded.
    filename : Pathlike
        The name of the file where the data is stored
    filter_keys : List[str] or Callable
        Optional argument to filter the keys in the returned object
    """
    with h5py.File(filename, "r") as h5f:
        keys = list(h5f.keys())
        if filter_keys is None:
            filter_keys = keys
        elif callable(filter_keys):
            filter_keys = [key for key in keys if filter_keys(key)]
        return {key: h5f[key][()] for key in keys if key in filter_keys}


def load_numpy(filename:Path, filter_keys:Optional[Union[List[str],Callable]]=None) -> DataDict:
    """
    Load the contents of a numpy compressed file as a dictionary where keys are the identifiers in the
    database and values are the contents. Takes optionally a `filter_keys` argument to control
    which keys are loaded.
    filename : Pathlike
        The name of the file where the data is stored
    filter_keys : List[str] or Callable
        Optional argument to filter the keys in the returned object
    """
    data = np.load(filename)
    if filter_keys is not None:
        if callable(filter_keys):
            filter_keys = [key for key in data if filter_keys(key)]
            data = {key: value for key, value in data if key in filter_keys}
    return data


class FeatLoader:

    def __init__(self, scpfile):

        with open(scpfile) as f:
            self.paths = dict(map(parse_line, f.readlines()))

        self._data = {}

    def _load(self, key):
        _, path = self.strip_path(self.paths[key])
        load = load_hdf5 if path.suffix == ".hdf5" else load_numpy
        self._data.update(load(path))

    def __getitem__(self, key):
        if key not in self.paths:
            raise KeyError(key)
        if key not in self._data:
            self._load(key)
        return self._data[key]

    def __len__(self):
        return len(self.paths)

    def to_dict(self):
        return {key: self[key] for key in self.paths}

    @staticmethod
    def strip_path(featpath):
        path, key = featpath.split(":")
        return key, Path(path)

