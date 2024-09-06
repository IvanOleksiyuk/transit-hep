"""Tests the function of kinematic_functions."""

import os

import jutils.data_loading as dl
import h5py
import numpy as np
import pytest


# create example hdf5 files for testing
num_files = 2
num_events = {0: 40, 1: 30}  # number of events per file
np.random.seed(42)

data_sets = {}
for i in range(num_files):
    jet1_obs = np.random.random(size=(num_events[i], 11)).astype(np.float32)
    jet2_obs = np.random.random(size=(num_events[i], 11)).astype(np.float32)
    jet1_cnsts = np.random.random(size=(num_events[i], 100, 4)).astype(np.float32)
    jet2_cnsts = np.random.random(size=(num_events[i], 100, 4)).astype(np.float32)
    data_sets[i] = {
        "jet1_obs": jet1_obs,
        "jet1_cnsts": jet1_cnsts,
        "jet2_obs": jet2_obs,
        "jet2_cnsts": jet2_cnsts,
    }


@pytest.fixture(scope="session")
def tmp_data(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data")
    files = []
    for i in data_sets:
        filename = os.path.join(fn, f"data_set_{i}.h5")
        files.append(filename)
        with h5py.File(filename, "w") as h5f:
            h5f.create_dataset("objects/jets/jet1_obs", data=data_sets[i]["jet1_obs"])
            h5f.create_dataset(
                "objects/jets/jet1_cnsts", data=data_sets[i]["jet1_cnsts"]
            )
            h5f.create_dataset("objects/jets/jet2_obs", data=data_sets[i]["jet2_obs"])
            h5f.create_dataset(
                "objects/jets/jet2_cnsts", data=data_sets[i]["jet2_cnsts"]
            )

    return files


##########################################
#         test load_rodem_data           #
##########################################


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading, expected_results",
    [
        (  ## General reading
            0,
            "all",
            -1,
            100,
            True,
            (data_sets[0]["jet1_obs"], data_sets[0]["jet1_cnsts"]),
        ),
        (  ## Selecting only 5 jets
            0,
            "all",
            5,
            100,
            True,
            (data_sets[0]["jet1_obs"][:5], data_sets[0]["jet1_cnsts"][:5]),
        ),
        (  ## Selecting subleading jets
            0,
            "all",
            -1,
            100,
            False,
            (data_sets[0]["jet2_obs"], data_sets[0]["jet2_cnsts"]),
        ),
        (  ## Only high level information
            0,
            "HL",
            -1,
            100,
            True,
            data_sets[0]["jet1_obs"],
        ),
        (  ## Only low level information
            0,
            "LL",
            -1,
            100,
            True,
            data_sets[0]["jet1_cnsts"],
        ),
        (  ## Selecting only 10 constituents
            0,
            "all",
            -1,
            10,
            True,
            (data_sets[0]["jet1_obs"], data_sets[0]["jet1_cnsts"][:, :10]),
        ),
        (  ## Selecting only 10 constituents but only high level
            0,
            "HL",
            -1,
            10,
            True,
            data_sets[0]["jet1_obs"],
        ),
        (  ## Selecting only 10 constituents but only low level
            0,
            "LL",
            -1,
            10,
            True,
            data_sets[0]["jet1_cnsts"][:, :10],
        ),
        ## The same as above but for two input files
        (  ## General reading
            -1,
            "all",
            -1,
            100,
            True,
            (
                np.concatenate(
                    [
                        data_sets[0]["jet1_obs"],
                        data_sets[1]["jet1_obs"],
                    ]
                ),
                np.concatenate(
                    [
                        data_sets[0]["jet1_cnsts"],
                        data_sets[1]["jet1_cnsts"],
                    ]
                ),
            ),
        ),
        (  ## Selecting only 10 jets (split 5 for each file)
            -1,
            "all",
            10,
            100,
            True,
            (
                np.concatenate(
                    [
                        data_sets[0]["jet1_obs"][:5],
                        data_sets[1]["jet1_obs"][:5],
                    ]
                ),
                np.concatenate(
                    [
                        data_sets[0]["jet1_cnsts"][:5],
                        data_sets[1]["jet1_cnsts"][:5],
                    ]
                ),
            ),
        ),
        (  ## Requesting more jets than are available
            -1,
            "all",
            1000,
            100,
            True,
            (
                np.concatenate(
                    [
                        data_sets[0]["jet1_obs"],
                        data_sets[1]["jet1_obs"],
                    ]
                ),
                np.concatenate(
                    [
                        data_sets[0]["jet1_cnsts"],
                        data_sets[1]["jet1_cnsts"],
                    ]
                ),
            ),
        ),
        (  ## Selecting subleading jets
            -1,
            "all",
            -1,
            100,
            False,
            (
                np.concatenate(
                    [
                        data_sets[0]["jet2_obs"],
                        data_sets[1]["jet2_obs"],
                    ]
                ),
                np.concatenate(
                    [
                        data_sets[0]["jet2_cnsts"],
                        data_sets[1]["jet2_cnsts"],
                    ]
                ),
            ),
        ),
        (  ## Only high level information
            -1,
            "HL",
            -1,
            100,
            True,
            np.concatenate(
                [
                    data_sets[0]["jet1_obs"],
                    data_sets[1]["jet1_obs"],
                ]
            ),
        ),
        (  ## Only low level information
            -1,
            "LL",
            -1,
            100,
            True,
            np.concatenate(
                [
                    data_sets[0]["jet1_cnsts"],
                    data_sets[1]["jet1_cnsts"],
                ]
            ),
        ),
        (  ## Selecting only 10 constituents
            -1,
            "all",
            -1,
            10,
            True,
            (
                np.concatenate(
                    [
                        data_sets[0]["jet1_obs"],
                        data_sets[1]["jet1_obs"],
                    ]
                ),
                np.concatenate(
                    [
                        data_sets[0]["jet1_cnsts"][:, :10],
                        data_sets[1]["jet1_cnsts"][:, :10],
                    ]
                ),
            ),
        ),
        (  ## Selecting only 10 constituents but only high level
            -1,
            "HL",
            -1,
            10,
            True,
            np.concatenate(
                [
                    data_sets[0]["jet1_obs"],
                    data_sets[1]["jet1_obs"],
                ]
            ),
        ),
        (  ## Selecting only 10 constituents but only low level
            -1,
            "LL",
            -1,
            10,
            True,
            np.concatenate(
                [
                    data_sets[0]["jet1_cnsts"][:, :10],
                    data_sets[1]["jet1_cnsts"][:, :10],
                ]
            ),
        ),
    ],
)
def test_load_rodem_data(
    filenumber, info, n_jets, n_cnsts, leading, expected_results, tmp_data
):
    """Tests whether load_rodem_data yields expected results."""
    if filenumber == -1:
        files = tmp_data
    else:
        files = tmp_data[filenumber]
    res = dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    np.testing.assert_equal(res, expected_results)


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading, expected_results",
    [
        (  ## One file, leading jet, too many constituents
            0,
            "LL",
            -1,
            110,
            True,
            np.concatenate(
                [
                    data_sets[0]["jet1_cnsts"],
                    np.zeros_like(data_sets[0]["jet1_cnsts"])[:, :10],
                ],
                axis=1,
            ),
        ),
        (  ## One file, sunleading jet, too many constituents
            0,
            "LL",
            -1,
            110,
            False,
            np.concatenate(
                [
                    data_sets[0]["jet2_cnsts"],
                    np.zeros_like(data_sets[0]["jet2_cnsts"])[:, :10],
                ],
                axis=1,
            ),
        ),
        (  ## Two files, leading jet, too many constituents
            -1,
            "LL",
            -1,
            110,
            True,
            np.concatenate(
                [
                    np.concatenate(
                        [
                            data_sets[0]["jet1_cnsts"],
                            np.zeros_like(data_sets[0]["jet1_cnsts"])[:, :10],
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            data_sets[1]["jet1_cnsts"],
                            np.zeros_like(data_sets[1]["jet1_cnsts"])[:, :10],
                        ],
                        axis=1,
                    ),
                ]
            ),
        ),
    ],
)
def test_load_rodem_data_check_zero_padding(
    filenumber, info, n_jets, n_cnsts, leading, expected_results, tmp_data
):
    """check correct zero-padding when requiring more cnsts than available."""
    if filenumber == -1:
        files = tmp_data
    else:
        files = tmp_data[filenumber]
    res = dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    np.testing.assert_equal(res, expected_results)


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading",
    [
        (0, "all", -1, 100, True),
        (1, "all", -1, 100, True),
    ],
)
def test_load_rodem_data_missing_files(
    filenumber, info, n_jets, n_cnsts, leading, tmp_data
):
    """check correct behaviour when encountering missing files."""
    if filenumber == 0:
        files = "missing_file"
    elif filenumber == 1:
        files = [tmp_data[0], "missing_file"]
    with pytest.raises(ValueError) as err:
        dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    assert str(err.value) == "Did not find missing_file."


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading",
    [
        (-1, "XYZ", -1, 100, True),
        (-1, 3, -1, 100, True),
    ],
)
def test_load_rodem_data_wrong_info(
    filenumber, info, n_jets, n_cnsts, leading, tmp_data
):
    """check correct behaviour when encountering wrong info."""
    if filenumber == -1:
        files = tmp_data
    else:
        files = tmp_data[filenumber]
    with pytest.raises(ValueError) as err:
        dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    assert (
        str(err.value)
        == f"Unknown info option {info}. Expected one of ['all', 'HL', 'LL']."
    )


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading",
    [
        (-1, "all", -1, [100], True),
        (-1, "all", -1, 100.0, True),
    ],
)
def test_load_rodem_data_wrong_n_cnsts_type(
    filenumber, info, n_jets, n_cnsts, leading, tmp_data
):
    """check correct behaviour when encountering wrong n_cnsts type."""
    if filenumber == -1:
        files = tmp_data
    else:
        files = tmp_data[filenumber]
    with pytest.raises(TypeError) as err:
        dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    assert (
        str(err.value) == f"n_cnsts has the wrong type {type(n_cnsts)}. Expected int."
    )


@pytest.mark.parametrize(
    "filenumber, info, n_jets, n_cnsts, leading",
    [
        (-1, "all", -1, 100, 1.0),
        (-1, "all", -1, 100, "true"),
    ],
)
def test_load_rodem_data_wrong_lead(
    filenumber, info, n_jets, n_cnsts, leading, tmp_data
):
    """check correct behaviour when encountering wrong type for leading."""
    if filenumber == -1:
        files = tmp_data
    else:
        files = tmp_data[filenumber]
    with pytest.raises(TypeError) as err:
        dl.load_rodem_data(files, info, n_jets, n_cnsts, leading)
    assert (
        str(err.value) == f"leading has the wrong type {type(leading)}. Expected bool."
    )
