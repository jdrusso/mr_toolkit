"""Main module with code for coarse-graining transition matrices and computing bin-weights."""
import numpy as np


def build_fine_transition_matrix(height_ratio: float, num_bins: int) -> np.ndarray:
    """
    Generate a Markov transition matrix where each bin is height_ratio more likely to transition to itself than to
    its neighbor.

    Parameters
    ----------
    height_ratio : float
        Ratio of the transition probability to self vs to neighbor bin.
        This is a proxy for the inter-bin barrier height.
    num_bins : int
        Number of bins in the transition matrix.

    Returns
    -------
    t_matrix : np.ndarray
        A (`num_bins` x `num_bins`) tri-diagonal, row-normalized transition matrix.
    """

    t_matrix = (
        np.eye(num_bins, num_bins) * height_ratio
        + np.eye(num_bins, num_bins, -1)
        + np.eye(num_bins, num_bins, 1)
    )

    normalized_t_matrix = t_matrix / np.sum(t_matrix, axis=1)[:, np.newaxis]

    return normalized_t_matrix


def compute_avg_bin_weights(initial_weights, transition_matrix, max_s: int, lag: int = 1):
    """
    Obtain the time-averaged bin weights for a lag of 1, described by

    .. math::  \\eqnwi

    Parameters
    ----------
    initial_weights : array-like
        List or array of initial microbin-weights.

    transition_matrix : np.ndarray
        (`n_states` x `n_states`) Transition matrix.

    max_s : int
        Maximum trajectory length :math:`S`

    lag : int
        Lag used for Markov model :math:`\lag`.

    Returns
    -------
    wi_bar : np.ndarray (`n_states`)
        List of time-averaged weights for each bin
    """

    assert max_s >= lag, "Trajectory length S is shorter than lag!"

    weights = np.full_like(initial_weights, fill_value=0.0)

    # Remember, at a lag of 1 this should iterate over values from 0 to max_s - 1
    # Need the +1 because range is end-exclusive
    for s in range(max_s - lag + 1):

        new_weights = np.dot(
            initial_weights, np.linalg.matrix_power(transition_matrix, s)
        )
        weights += new_weights

    weights /= (max_s - lag + 1)

    return weights


def coarse_grain(P: np.ndarray, cg_map: np.ndarray, w: np.ndarray, lag: int = 1, normalize: bool = True) -> np.ndarray:
    """
    Coarse-grains a fine-grained transition matrix according to some mapping of microstates to macrostates and weights
    over the microstates.

    This is done according to

    .. math:: \eqncg

    Parameters
    ----------
    P : np.ndarray
        Fine-grained transition matrix.
    cg_map : list of lists
        List of all microstates in each macrostate.
    w : array-like
        Microbin weights :math:`\\wi`.
    lag : int
        Lag for Markov model :math:`\\lag`.
    normalize : bool
        Normalize the resulting matrix over the weights.
        This should be off when building an occupancy matrix over many lags, because there the normalization is over
        all :math:`\\wi`.

    Returns
    -------
    p_matrix : np.ndarray
        Coarse-grained transition matrix :math:`\\textbf{T}`.

    Examples
    --------
    To coarse-grain a 6x6 transition matrix P into a 4x4 by grouping the inner pairs of states (1+2 and 3+4) and leaving
    the edge states unchanged, one could do

    >>> coarse_grain(P, [[0], [1,2], [2,3], [4]], w)
    """

    num_cg_bins = len(cg_map)

    T = np.full(shape=(num_cg_bins, num_cg_bins), fill_value=0.0)

    # Iterate over every pair of n,m
    for m in range(num_cg_bins):
        for n in range(num_cg_bins):

            # For each of those pairs, iterate over each of the i, j elements
            for i in cg_map[m]:
                for j in cg_map[n]:

                    T[m, n] += w[i] * np.linalg.matrix_power(P, lag)[i, j]

            # Finished an m,n pair, so normalize by the total weight of macrobin m
            microbins = cg_map[m]
            w_tot = np.sum(w[microbins])

            if normalize:
              T[m, n] /= w_tot

    return T


def build_occupancy(fg_matrix: np.ndarray, initial_weights: np.ndarray, cg_map: list, s: int, time_horizon: int) -> np.ndarray:
    """
    Builds the occupancy matrix as

    .. math::  \\eqnbuildocc

    Parameters
    ----------
    fg_matrix : np.ndarray
        The fine-grained matrix :math:`\\Tfg`.
    initial_weights : np.ndarray or list
        Vector of initial weights :math:`\\wi`.
    cg_map : list of lists
        List of all microstates in each macrostate.
    s : int
        Maximum trajectory length :math:`S`.
    time_horizon : int
        Time horizon :math:`TH`.

    Returns
    -------
    occ : np.ndarray
        The occupancy matrix, computed as above.

    Todo
    ----
    Rather than explicitly row normalizing, store all the weights, and then normalize by them in the "correct" way.

    """

    n_cg_bins = len(cg_map)
    occupancy = np.zeros(shape=(n_cg_bins, n_cg_bins))

    for lag in range(1, time_horizon+1):

        w_i = compute_avg_bin_weights(initial_weights, fg_matrix, max_s=s, lag=lag)

        cg_matrix = coarse_grain(fg_matrix, cg_map, w_i, lag=lag, normalize=False)
        occupancy += cg_matrix

    occupancy /= time_horizon
    normed_occupancy = occupancy / np.sum(occupancy, axis=1)[:, np.newaxis]

    return normed_occupancy
