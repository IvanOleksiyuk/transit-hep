"""Provides functions to calculate kinematic observables."""


from typing import Union

import numpy as np


def invariant_mass(jets: np.ndarray) -> np.ndarray:
    """Calculate the invariant mass of a collection of 4-vectors.

    Take an array of the shape (n_vectors, 4) where the 4 columns correspond to (E, px,
    py, pz) and calculate the invariant mass as m = sqrt(E^2 - px^2 - py^2 - pz^2).

    Args:
        jets: 2-dim ndarray of shape (n_vectors, 4)

    Returns:
        invM: masked array of length n_vectors
    """
    if not isinstance(jets, np.ndarray):
        raise TypeError("jets must be of type 'np.ndarray'")
    if jets.ndim != 2:
        raise ValueError(
            f"Expected an array of dimension 2 but received dimension {jets.ndim}."
        )
    if jets.shape[1] != 4:
        raise ValueError(
            f"Expected an array of shape (_, 4) but received {jets.shape}."
        )
    squared = jets**2
    return np.ma.masked_invalid(
        np.sqrt(squared[:, 0] - squared[:, 1] - squared[:, 2] - squared[:, 3])
    )


def delta_r(
    eta1: Union[float, np.float64, np.ndarray],
    eta2: Union[float, np.float64, np.ndarray],
    phi1: Union[float, np.float64, np.ndarray],
    phi2: Union[float, np.float64, np.ndarray],
) -> Union[float, np.float64, np.ndarray]:
    """Calculates delta_r values between given ndarrays.

    Calculates the delta_r between objects. Takes unstructed ndarrays (or
    scalars) of eta1, eta2, phi1 and phi2 as input. Returns in the same format.
    This function can either handle eta1, phi1 to be numpy arrays (and eta2,
    phi2 to be floats), eta2, phi2 to be numpy arrays (and eta1, phi1 to be
    floats), or all four to be floats, and all four to be numpy arrays.
    Whenever numpy arrays are involved, they must be one-dimensional and of the
    same length.

    Args:
        eta1: First unstructured ndarray of eta values.
        eta2: Second unstructured ndarray of eta values.
        phi1: First unstructured ndarray of phi values.
        phi2: Second unstructured ndarray of phi values.

    Returns:
        Unstructured array of delta_r values between the pairs.

    """
    if type(eta1) is not type(phi1):
        raise TypeError("Inputs 'eta1' and 'phi1' must be of the same type")
    if type(eta2) is not type(phi2):
        raise TypeError("Inputs 'eta2' and 'phi2' must be of the same type")
    if not isinstance(eta1, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if not isinstance(eta2, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(eta1, np.ndarray) and isinstance(phi1, np.ndarray):
        if eta1.ndim != 1 or phi1.ndim != 1:
            raise TypeError("Dimension of 'eta1' or 'phi1' is not equal to 1")
        if len(eta1) != len(phi1):
            raise TypeError("Lengths of 'eta1' and 'phi1' do not match")
    if isinstance(eta2, np.ndarray) and isinstance(phi2, np.ndarray):
        if eta2.ndim != 1 or phi2.ndim != 1:
            raise TypeError("Dimension of 'eta2' or 'phi2' is not equal to 1")
        if len(eta2) != len(phi2):
            raise TypeError("Lengths of 'eta2' and 'phi2' do not match")
    if (
        isinstance(eta1, np.ndarray)
        and isinstance(eta2, np.ndarray)
        and len(eta1) != len(eta2)
    ):
        raise TypeError(
            "If 'eta1', 'eta2', 'phi1', 'phi2' are all of type np.ndarray, "
            "their lengths must be the same"
        )

    deta = np.absolute(eta1 - eta2)
    dphi = np.absolute(phi1 - phi2) % (2 * np.pi)
    dphi = np.min([2 * np.pi - dphi, dphi], axis=0)
    return np.sqrt(deta * deta + dphi * dphi)
