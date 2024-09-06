"""Tests the function of kinematic_functions."""

import numpy as np
import pytest
import jutils.kinematic_functions as kf

##########################################
#            test delta_r                #
##########################################


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2, expected_results",
    [
        (0.0, 1.0, 0.0, 1.0, np.sqrt(2)),
        (1.0, 0.0, 1.0, 0.0, np.sqrt(2)),
        (0.0, 1.0, 1.0, 0.0, np.sqrt(2)),
        (1.0, 0.0, 0.0, 1.0, np.sqrt(2)),
        (0.0, 0.0, 0.0, -2.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, 0.0, -1.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, 0.0, -0.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, 0.0, 0.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, 0.0, 1.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, 0.0, 2.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, -2.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, -1.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, -0.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, 0.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi),
        (0.0, 0.0, np.pi, 2.5 * np.pi, 0.5 * np.pi),
        (
            np.array([0.0, 1.0]),
            0.0,
            np.array([0.0, 0.0]),
            1.0,
            np.array([1.0, np.sqrt(2)]),
        ),
        (
            0.0,
            np.array([0.0, 1.0]),
            1.0,
            np.array([0.0, 0.0]),
            np.array([1.0, np.sqrt(2)]),
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, np.sqrt(2), np.sqrt(5)]),
        ),
    ],
)
def test_delta_r(eta1, eta2, phi1, phi2, expected_results):
    """Tests whether delta_r calculation yields expected results."""
    res = kf.delta_r(eta1, eta2, phi1, phi2)
    np.testing.assert_allclose(res, expected_results)


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (1, 1, 1, 1),
        ("1", "1", "1", "1"),
        (True, True, True, True),
        (1.0, 1, 1.0, 1),
        (1, 1.0, 1, 1.0),
    ],
)
def test_delta_r_wrong_type(eta1, eta2, phi1, phi2):
    """Tests whether correct typing is enforced.

    We only foresee this function to take float, np.float64 or numpy ndarray.
    All other input types should be caught.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert (
        str(err.value) == "Inputs must be of type 'float', 'np.float64' or 'np.ndarray'"
    )


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (1.0, 1.0, 1, 1.0),
        (1.0, 1.0, np.array([1]), 1.0),
        (1.0, 1.0, [1.0, 1.0], 1.0),
        (1, 1.0, 1.0, 1.0),
        (np.array([1]), 1.0, 1.0, 1.0),
        ([1.0, 1.0], 1.0, 1.0, 1.0),
    ],
)
def test_delta_r_mixed_type1(eta1, eta2, phi1, phi2):
    """Tests whether mixed types (1) are caught.

    eta1 and phi1 must be of the same type, otherwise an exception should be
    raised. This function tests for that exception.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Inputs 'eta1' and 'phi1' must be of the same type"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (1.0, 1.0, 1.0, 1),
        (1.0, 1.0, 1.0, np.array([1])),
        (1.0, 1.0, 1.0, [1.0, 1.0]),
        (1.0, 1, 1.0, 1.0),
        (1.0, np.array([1]), 1.0, 1.0),
        (1.0, [1.0, 1.0], 1.0, 1.0),
    ],
)
def test_delta_r_mixed_type2(eta1, eta2, phi1, phi2):
    """Tests whether mixed types (1) are caught.

    eta1 and phi1 must be of the same type, otherwise an exception should be
    raised. This function tests for that exception.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Inputs 'eta2' and 'phi2' must be of the same type"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (np.ones((2, 1)), 1.0, np.ones(1), 1.0),
        (np.ones(1), 1.0, np.ones((2, 1)), 1.0),
    ],
)
def test_delta_r_dim_mismatch1(eta1, eta2, phi1, phi2):
    """Tests whether exception is raised if there are dimension mismatches.

    If 'eta1' and 'phi1' are of type numpy ndarray, their dimensions must be
    the same, otherwise an exception should be raised. They can also only be 1D
    arrays.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Dimension of 'eta1' or 'phi1' is not equal to 1"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (1.0, np.ones((2, 1)), 1.0, np.ones(1)),
        (1.0, np.ones(1), 1.0, np.ones((2, 1))),
    ],
)
def test_delta_r_dim_mismatch2(eta1, eta2, phi1, phi2):
    """Tests whether exception is raised if there are dimension mismatches.

    If 'eta2' and 'phi2' are of type numpy ndarray, their dimensions must be
    the same, otherwise an exception should be raised. They can also only be 1D
    arrays.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Dimension of 'eta2' or 'phi2' is not equal to 1"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (np.ones(2), 1.0, np.ones(1), 1.0),
        (np.ones(1), 1.0, np.ones(2), 1.0),
    ],
)
def test_delta_r_length_mismatch1(eta1, eta2, phi1, phi2):
    """Test whether exception is raised if length of arrays mismatch.

    If 'eta1' and 'phi1' are of type numpy ndarray, once confirmed their
    dimensions are the same, their 1D lengths must match, otherwise an
    exception should be raised.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Lengths of 'eta1' and 'phi1' do not match"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (1.0, np.ones(2), 1.0, np.ones(1)),
        (1.0, np.ones(1), 1.0, np.ones(2)),
    ],
)
def test_delta_r_length_mismatch2(eta1, eta2, phi1, phi2):
    """Test whether exception is raised if length of arrays mismatch.

    If 'eta2' and 'phi2' are of type numpy ndarray, once confirmed their
    dimensions are the same, their 1D lengths must match, otherwise an
    exception should be raised.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert str(err.value) == "Lengths of 'eta2' and 'phi2' do not match"


@pytest.mark.parametrize(
    "eta1, eta2, phi1, phi2",
    [
        (np.ones(5), np.ones(3), np.ones(5), np.ones(3)),
        (np.ones(1), np.ones(3), np.ones(1), np.ones(3)),
    ],
)
def test_delta_r_all_length_mismatch(eta1, eta2, phi1, phi2):
    """Tests whether exception is raised if arrays are not of equal length.

    If all of 'eta1', 'eta2', 'phi1', 'phi2' are of type numpy ndarray, and
    all of them are 1-dimensional, their lengths must be the same to guarantee
    that the operations in the delta_r calculation are performed element-wise.
    """
    with pytest.raises(TypeError) as err:
        kf.delta_r(eta1, eta2, phi1, phi2)
    assert (
        str(err.value) == "If 'eta1', 'eta2', 'phi1', 'phi2' are all of type "
        "np.ndarray, their lengths must be the same"
    )


##########################################
#          test invariant_mass           #
##########################################


@pytest.mark.parametrize(
    "jets, expected_results",
    [
        (np.array([[1.0, 0.0, 0.0, 0.0]]), np.array([1.0])),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0, 0],
                    [1.0, 0.0, 0.0, 1.0],
                    [5.0, 4.0, 0.0, 0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.array([1.0, 0.0, 0.0, 0.0, 3.0, 0.0]),
        ),
    ],
)
def test_invariant_mass(jets, expected_results):
    """Tests whether invariant_mass calculation yields expected results."""
    res = kf.invariant_mass(jets)
    np.testing.assert_allclose(res, expected_results)


@pytest.mark.parametrize(
    "jets, expected_results",
    [
        (
            np.array([[0.0, 1.0, 0.0, 0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1]]),
            np.array([np.nan] * 3),
        )
    ],
)
def test_invariant_mass_negative_mass_squared(jets, expected_results):
    """Test for correct behaviour when unphysical values are passed."""
    with pytest.warns(
        RuntimeWarning, match="invalid value encountered in sqrt"
    ) as warn:
        res = kf.invariant_mass(jets)
    np.testing.assert_array_equal(res.data, expected_results)


@pytest.mark.parametrize(
    "jets", [([1, 1, 1, 1]), ([[1, 1, 1, 1], [2, 2, 2, 2]]), (1.0), (1)]
)
def test_invariant_mass_type(jets):
    """Tests whether a TypeError is raised when the false input type is used."""
    with pytest.raises(TypeError) as err:
        kf.invariant_mass(jets)
    assert str(err.value) == "jets must be of type 'np.ndarray'"


@pytest.mark.parametrize(
    "jets, ndim",
    [
        (np.ones((2, 4, 2)), 3),
        (np.ones(4), 1),
    ],
)
def test_invariant_mass_dim_mismatch2(jets, ndim):
    """Tests whether exception is raised if there are dimension mismatches.

    jets must be a 2-dim array.
    """
    with pytest.raises(ValueError) as err:
        kf.invariant_mass(jets)
    assert (
        str(err.value)
        == f"Expected an array of dimension 2 but received dimension {ndim}."
    )


@pytest.mark.parametrize(
    "shape",
    [
        ((2, 3)),
        ((2, 5)),
    ],
)
def test_invariant_mass_shape_mismatch(shape):
    """Tests whether exception is raised if there are shape mismatches.

    jets must be of shape (_, 4).
    """
    with pytest.raises(ValueError) as err:
        kf.invariant_mass(np.ones(shape))
    assert str(err.value) == f"Expected an array of shape (_, 4) but received {shape}."
