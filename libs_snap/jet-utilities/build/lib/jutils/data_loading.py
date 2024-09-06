"""Functions for data loading.

This module contains functions to load data from the RODEM data set. While
load_rodem_date just reads the data from one or several files and concatenates it,
read_all_data uses a yaml file to read a signal, background, and hold-out data set.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict
import importlib_resources as pkg_resources

import cerberus
import h5py
import numpy as np
import yaml


def div_parts(a: int, b: int) -> list:
    """Divides a by b, returning a list with b nearly equal rounded intergers which
    still sums to a"""
    q, r = divmod(a, b)
    return r * [q + 1] + (b - r) * [q]


def load_rodem_data(
    files: Union[str, List[str]],
    info: str = "all",
    n_jets: int = -1,
    n_cnsts: int = -1,
    leading: bool = True,
    incl_substruc: bool = True,
    astype: Union[type, str] = np.float32,
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """Read rodem data into np.ndarray.

    Load the datasets with the constituent particles and the high-level jet observables
    from RODEM HDF5 files.

    Args:
        files: path or list of paths to HDF5 files with the events
        info:   Which information of the input files to return. Possible values are
                "all", "HL", "LL" reffering to everything, high-level information of the
                jets, and constituents, respectively.
        n_cnsts: Number of constituents to consider. When n_cnsts==-1, it will be
            automatically be set to the shape of the first file
        leading: Load the leading jet from each event. If False subleading is loaded.
        n_jets: The total number of jets to load
            Will be divided as equally as possible for each file passed
        incl_substruc: Include jet substructure vars in the high-level array
        astype: The dtype to use for the data represented as a string or npdtype

    Returns:
        One or two ndarrays (depending on the info argument)
        These arrays will contain the kinematics [pt, eta, phi, M]
        of the jets and constituents, which will be 0 padded.

        Depending on the incl_substruc argument, the obs ndarray may also include
        substructure variables.
    """
    # make sure that all the inputs make sense
    if isinstance(files, (str, Path)):
        files = [files]
    for file_ in files:
        if not os.path.isfile(file_):
            raise ValueError(f"Did not find {file_}.")
    if info not in ["all", "HL", "LL"]:
        raise ValueError(
            f"Unknown info option {info}. Expected one of ['all', 'HL', 'LL']."
        )
    if not isinstance(n_cnsts, int):
        raise TypeError(f"n_cnsts has the wrong type {type(n_cnsts)}. Expected int.")
    if not isinstance(leading, bool):
        raise TypeError(f"leading has the wrong type {type(leading)}. Expected bool.")
    if not isinstance(n_jets, int):
        raise TypeError(f"n_jets has the wrong type {type(n_jets)}. Expected int.")
    if n_cnsts > 0 and info == "HL":
        print(
            "Warning: Specified the number of constituents but requested only HL"
            " observables."
        )

    # declare the variables that are requested
    requested_info = set()
    if info == "HL":
        requested_info.add("obs")
    elif info == "LL":
        requested_info.add("cnsts")
    elif info == "all":
        requested_info.add("cnsts")
        requested_info.add("obs")

    # The flags to use to read from the datatable
    obs_flag = f"jet{1 if leading else 2}_obs"
    cst_flag = f"jet{1 if leading else 2}_cnsts"

    # the number of high level variables to read, without substruc it is pt,eta,phi,M
    obs_dim = -1 if incl_substruc else 4

    # We want to pre allocate the arrays. This is to reduce memory overhead and time
    # used by the np.concatenate function which momentarily doubles the memory used.
    # For large n_cnsts and large n_jets this is quite an issue!
    # Even the extra loop of calculating the lengths of each file is comparively small!

    # how many jets or constituents are available in each file
    nj_pf = div_parts(n_jets, len(files)) if n_jets != -1 else len(files) * [np.inf]
    nc_pf = len(files) * [n_cnsts] if n_cnsts != -1 else len(files) * [np.inf]
    for i, file_ in enumerate(files):

        # check the size of the data in the file
        with h5py.File(file_, "r") as hf:
            file_jets = len(hf["objects/jets/jet1_obs"])
            file_csts = hf["objects/jets/jet1_cnsts"].shape[1]
            cst_dim = hf["objects/jets/jet1_cnsts"].shape[-1]

            # if using the substrucutre variables then set the obs dim to be higher
            if incl_substruc:
                obs_dim = hf["objects/jets/jet1_obs"].shape[1]

        # if we requested more jets than what is available, only read those and warn
        if n_jets != -1 and file_jets < nj_pf[i]:
            print(
                f"Warning: Requested {nj_pf[i]} jets from {Path(file_).name} "
                f"which only has {file_jets}. Resulting array will be smaller."
            )
        nj_pf[i] = min(nj_pf[i], file_jets)

        # if we requested more constituents than available, only read those and warn
        if n_cnsts != -1 and file_csts < nc_pf[i]:
            print(
                f"Warning: Requested {nc_pf[i]} constituents from {Path(file_).name} "
                f"which only has {file_csts}. May lead to excessive padding."
            )
        nc_pf[i] = min(nc_pf[i], file_csts)

    # if using auto n_csts, use the max from all of the files
    if n_cnsts == -1:
        n_cnsts = max(nc_pf)

    # declare the arrays of the appropriate shapes and dtype
    obs = np.zeros((sum(nj_pf) * ("obs" in requested_info), obs_dim), dtype=astype)
    cnsts = np.zeros(
        (sum(nj_pf) * ("cnsts" in requested_info), n_cnsts, cst_dim), dtype=astype
    )

    # fill in the declared arrays file by file
    rc = 0  # running count of the jets in the array
    for nj, nc, file_ in zip(nj_pf, nc_pf, files):
        with h5py.File(file_, "r") as hf:
            if "obs" in requested_info:
                obs[rc : rc + nj] = hf[f"objects/jets/{obs_flag}"][:nj, :obs_dim]
            if "cnsts" in requested_info:
                cnsts[rc : rc + nj, :nc] = hf[f"objects/jets/{cst_flag}"][:nj, :nc]
            rc += nj

    # return the appropriate information
    if info == "all":
        return obs, cnsts
    if info == "HL":
        return obs
    if info == "LL":
        return cnsts
    raise RuntimeError("Not clear which information to return")


def read_all_data(config: Union[dict, str]) -> Dict[str, Union[Tuple, np.ndarray]]:
    """Read data and return it as dict of ndarrays or dict of tuples of ndarrays

    Reads data using load_rodem_data() and a dictionary of settings which is
    validated using cerberus and the schema: data_loading_schema.yml

    Args:
        config: Dict or string containing info and kwargs for load_rodem_data which
            defined what data is returned. If argument is a string then it must be a
            path to a yaml file containing the dict
    Returns:
        dictionary of np.ndarrays, the keys are specified by the config dict.
    """
    # if the config is a path then we must load the yaml file it points to
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    # read and validate settings (if from a loaded package we need to use open text)
    if __package__ == "":
        schema_file = pkg_resources.open_text("jutils", "data_loading_schema.yml")
    else:
        schema_file = pkg_resources.open_text(__package__, "data_loading_schema.yml")
    schema = yaml.load(schema_file, Loader=yaml.FullLoader)
    schema_file.close()
    validator = cerberus.Validator(schema)
    if validator.validate(config):
        print("Validated settings.")
    else:
        print(validator.errors)
        sys.exit(1)

    # read data sets
    datasets = {}
    for set_ in config["data_sets"].keys():
        datasets[set_] = load_rodem_data(
            config["data_sets"][set_], **config["data_type"]
        )

        # print the number of data points loaded
        if isinstance(datasets[set_], tuple):
            n_loaded = len(datasets[set_][0])
        else:
            n_loaded = len(datasets[set_])
        print(f"read {n_loaded} jets into set {set_}")

    return datasets


if __name__ == "__main__":
    read_all_data("data_loading_example.yml")
