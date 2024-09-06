"""Provides utility functions for coordinate transformations.

Various functions are provided to perform coordinate transformations and
pre-processing steps to the constituent particles. Retains standard formats of
either (pt, eta, phi, m) ["ATLAS coordiantes"] or (E, px, py, pz) ["Cartesian
coordinates"]. Other orderings etc should be totally avoided for consistency.

"""

import numpy as np


def boost_constituents(jets, cnsts, m_pt=0.5):
    """Returns constituents of jets boosted to a fixed m/pt.

    Given an ndarray of jets and an ndarray of their associated constituent
    particles, apply a boost to the constituent particles in such a way that the
    m/pt of each jet reaches the pre-defined value.

    Args:
        jet: An ndarray of jets in (pt, eta, phi, m) coordinates.
        cnsts: An ndarray corresponding to the constituent particles of the jets,
          given in shape (n_jets, n_cnsts, 4) with the latter dimension being the
          (pt, eta, phi, m) coordinates.
        m_pt: A float with the desired m/pt value.

    Returns: An ndarray of the constituents boosted in such a way that each jet
      reaches the pre-defined m/pt value.

    """

    def get_boost_vector(pt, eta, phi, m):
        """Returns boost vectors in x, y, z coordinates."""
        x = pt * np.cos(phi)
        y = pt * np.sin(phi)
        z = pt * np.sinh(eta)
        p2 = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
        e = np.sqrt(p2 + np.power(m, 2))
        return np.concatenate(
            [
                np.divide(x, e).reshape(-1, 1),
                np.divide(y, e).reshape(-1, 1),
                np.divide(z, e).reshape(-1, 1),
            ],
            axis=-1,
        )

    def apply_boost(c, b):
        """Boosts a list of constituents and returns it.

        Takes a list of constituents of shape (n_jets, n_cnsts, 4) in
        coordinates (E, px, py, pz) and boosts it along a given list of length
        n_jets of boost 3-vectors with components (bx, by, bz).

        """
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
        b2 = np.power(bx, 2) + np.power(by, 2) + np.power(bz, 2)
        gamma = np.ones(shape=b2.shape) / np.sqrt(1.0 - b2)
        gamma2 = np.where(b2 > 0.0, (gamma - 1.0) / b2, 0)

        bx = np.repeat(bx, c.shape[1]).reshape(-1, c.shape[1])
        by = np.repeat(by, c.shape[1]).reshape(-1, c.shape[1])
        bz = np.repeat(bz, c.shape[1]).reshape(-1, c.shape[1])
        gamma = np.repeat(gamma, c.shape[1]).reshape(-1, c.shape[1])
        gamma2 = np.repeat(gamma2, c.shape[1]).reshape(-1, c.shape[1])

        bp = bx * c[:, :, 1] + by * c[:, :, 2] + bz * c[:, :, 3]

        c[:, :, 1] = c[:, :, 1] + gamma2 * bp * bx - gamma * bx * c[:, :, 0]
        c[:, :, 2] = c[:, :, 2] + gamma2 * bp * by - gamma * by * c[:, :, 0]
        c[:, :, 3] = c[:, :, 3] + gamma2 * bp * bz - gamma * bz * c[:, :, 0]
        c[:, :, 0] = gamma * (c[:, :, 0] - bp)
        return c

    boost_vectors1 = get_boost_vector(jets[:, 0], jets[:, 1], jets[:, 2], jets[:, 3])
    boost_vectors2 = get_boost_vector(
        -jets[:, 3] / m_pt, jets[:, 1], jets[:, 2], jets[:, 3]
    )

    cnsts = to_cartesian(cnsts)
    cnsts = apply_boost(cnsts, boost_vectors1)
    cnsts = apply_boost(cnsts, boost_vectors2)
    cnsts = to_atlas(cnsts)
    cnsts = np.ma.array(cnsts, mask=cnsts.mask)

    return cnsts


def to_atlas(cnsts):
    """Converts jet constituents into ATLAS coordinates.

    This converts a collection of jet constituent fourvectors from an (E, px,
    py, pz) representation to (pt, eta, phi, m). The input array of fourvectors
    must be of shape (n_jets, n_cnsts, 4) where n_cnsts is a fixed number of
    constituent particles considered. Returns the same shape, but with
    fourvectors in ATLAS coordinates.

    TODO: properly take care of the eta = 0 case.

    TODO: consider switching to an implementation, where phi is defined to be in
    [0, 2pi). Implementation: phi = phi % (2 * np.pi)

    Args:
        cnsts: a numpy ndarray of jet constituent particles of shape
          (n_jets, n_cnsts, 4) and in Cartesian coordinates.

    Returns: the same fourvectors in ATLAS coordinates.

    """
    n_cnsts = cnsts.shape[1]
    mask = cnsts.mask

    pt = np.sqrt(np.sum(np.power(cnsts[:, :, 1:3], 2), axis=-1))
    p = np.sqrt(np.sum(np.power(cnsts[:, :, 1:4], 2), axis=-1))

    eta = -0.5 * np.log((1 - cnsts[:, :, 3] / p) / (1 + cnsts[:, :, 3] / p))
    try:
        eta[(cnsts[:, :, -1].mask) | (cnsts[:, :, -1] == 0)] = 0
    except:
        eta[(cnsts[:, :, -1] == 0)] = 0

    phi = np.arctan2(cnsts[:, :, 2], cnsts[:, :, 1])
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    phi = np.where(phi <= -np.pi, phi + 2 * np.pi, phi)

    e2 = np.power(cnsts[:, :, 0], 2)
    p2 = np.power(p[:, :], 2)
    m = np.where((e2 - p2) > 0, np.sqrt(e2 - p2), 0)

    return np.ma.array(
        np.concatenate(
            [
                pt.reshape(-1, n_cnsts, 1),
                eta.reshape(-1, n_cnsts, 1),
                phi.reshape(-1, n_cnsts, 1),
                m.reshape(-1, n_cnsts, 1),
            ],
            axis=-1,
        ),
        mask=cnsts.mask,
    )


def to_cartesian(cnsts):
    """Converts jet constituents into Cartesian coordinates.

    This converts a collection of jet constituent fourvectors from an (pt, eta,
    phi, m) representation to (E, px, py, pz). The input array of fourvectors
    must be of shape (n_jets, n_cnsts, 4) where n_cnsts is a fixed number of
    constituent particles considered. Returns the same shape, but with
    fourvectors in Cartesian coordinates.

    TODO: fix possible rounding issues with energy calculation.

    Args:
        cnsts: a numpy ndarray of jet constituent particles of shape (n_jets,
          n_cnsts, 4) and in ATLAS coordinates.

    Returns: the same fourvectors in Cartesian coordinates.

    """
    n_cnsts = cnsts.shape[1]
    px = cnsts[:, :, 0] * np.cos(cnsts[:, :, 2])
    py = cnsts[:, :, 0] * np.sin(cnsts[:, :, 2])
    pz = cnsts[:, :, 0] * np.sinh(cnsts[:, :, 1])

    p2 = np.power(px[:, :], 2) + np.power(py[:, :], 2) + np.power(pz[:, :], 2)
    e = np.sqrt(p2 + np.power(cnsts[:, :, 3], 2))

    return np.ma.concatenate(
        [
            e.reshape(-1, n_cnsts, 1),
            px.reshape(-1, n_cnsts, 1),
            py.reshape(-1, n_cnsts, 1),
            pz.reshape(-1, n_cnsts, 1),
        ],
        axis=-1,
    )


def to_cartesian_2d(jets):
    """Wrapper for to_cartesian.

    to_cartesian takes and returns an 3-dimensional array of shape (n_jets, n_cnsts, 4)
    whereas one might have and want an 2-dimensional array of shape (n_4vectors, 4).
    This wrapper takes care of that.

    Args:
        jets: ndarray of shape (num_jets, 4) containing (pT, eta, phi, m) in this order

    Returns:
        cartesian: ndarray of the same shape as jets but with the coordinates
            transformed to (E, px, py, pz).
    """
    if jets.ndim != 2:
        raise ValueError(
            f"Expected an array of dimension 2 but received dimension {jets.ndim}."
        )
    if jets.shape[1] != 4:
        raise ValueError(
            f"Expected an array of shape (_, 4) but received {jets.shape}."
        )
    return to_cartesian(jets.reshape(-1, 1, 4)).reshape(-1, 4)


def topo_dnn_centering(cnsts):
    """Centers constituents as done for the topo-DNN top tagger.

    Given a masked ndarray of constituent particles (in ATLAS coordinates),
    calculates the positions of all constituent particles with respect to the
    first (leading in pT) constituent, and rotates them around that leading
    constituent in such a way that the second constituent aligns with phi = 0.
    Flips jet if the pT-weighted sum over all phis is negative. Then returns the
    constituents in the new coordinates as a masked ndarray.

    For details, see: https://arxiv.org/abs/1704.02124

    Input must be of shape ((n_jets, n_cnsts, 4), n_jets, n_jets), where the 4
    entries per constituent are given in (pt, eta, phi, m). Output will be an
    ndarray of the constituents of the same shape.

    Args:
        cnsts: A masked ndarray of constituent particles in ATLAS coordinates.

    Returns: A masked ndarray of the transformed constituent particles.

    """
    n_cnsts = cnsts.shape[1]

    cnsts[:, :, 1] -= cnsts[:, 0, 1].reshape(-1, 1)
    cnsts[:, :, 2] -= cnsts[:, 0, 2].reshape(-1, 1)

    def shift_to_plus_minus_pi(a):
        """Shifts the given array to values between -pi and pi."""
        a = np.where(a < np.pi, a + 2 * np.pi, a)
        a = np.where(a >= np.pi, a - 2 * np.pi, a)
        return a

    cnsts[:, :, 2] = shift_to_plus_minus_pi(cnsts[:, :, 2])

    alpha = -np.arctan2(cnsts[:, 1, 2], cnsts[:, 1, 1])
    alpha = np.repeat(alpha, n_cnsts).reshape(-1, n_cnsts)
    etas = cnsts[:, :, 1] * np.cos(alpha) - cnsts[:, :, 2] * np.sin(alpha)
    phis = cnsts[:, :, 1] * np.sin(alpha) + cnsts[:, :, 2] * np.cos(alpha)

    cnsts[:, :, 1] = etas
    cnsts[:, :, 2] = phis

    flip = np.sum((cnsts[:, :, 2] * cnsts[:, :, 0]), axis=-1) < 0
    cnsts[:, :, 2] = np.ma.where(
        np.repeat(flip, n_cnsts).reshape(-1, n_cnsts),
        -1 * cnsts[:, :, 2],
        cnsts[:, :, 2],
    )

    return cnsts
