"""Main module."""
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
        A num_bins x num_bins tri-diagonal, row-normalized transition matrix.
    """

    t_matrix = (
        np.eye(num_bins, num_bins) * height_ratio
        + np.eye(num_bins, num_bins, -1)
        + np.eye(num_bins, num_bins, 1)
    )

    normalized_t_matrix = t_matrix / np.sum(t_matrix, axis=1)[:, np.newaxis]

    return normalized_t_matrix


def coarse_grain(P: np.ndarray, cg_map: np.ndarray, w: np.ndarray, lag: int = 1, normalize=True):
    """
    Coarse-grains a fine-grained transition matrix according to some mapping of microstates to macrostates and weights
    over the microstates.

    Parameters
    ----------
    P : np.ndarray
        Fine-grained transition matrix.
    cg_map : list of lists
        List of all microstates in each macrostate.
    w : np.ndarray
        Microbin weights.
    lag : int
        Lag for Markov model.
    normalize : bool
        Normalize the resulting matrix over the weights.
        This should be off when building an occupancy matrix over many lags, because there the normalization is over
            all w_i.

    Returns
    -------
    p_matrix : np.ndarray
        Coarse-grained transition matrix.

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


def compute_avg_bin_weights(initial_weights, transition_matrix, max_s: int, lag: int = 1):
    """
    Obtain the time-averaged bin weights for a lag of 1, described by

    .. math::  \\bar{w_i} = \\frac{1}{S} \\sum_{s=0}^{S-1} \\sum_k w_k(0) \\, (\\mathbf{P}^s)_{k \\rightarrow i}

    Parameters
    ----------
    initial_weights : np.ndarray or list
        List or array of initial microbin-weights.

    transition_matrix : np.ndarray
        Transition matrix.

    max_s : int
        Time horizon S

    lag : int
        Lag used for Markov model.

    Returns
    -------
    wi_bar : np.ndarray
        List of time-averaged weights for each bin
    """

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


def build_occupancy(fg_matrix: np.ndarray, initial_weights: np.ndarray, cg_map, s, time_horizon: int):
    """
    Builds the occupancy matrix as

    .. math::  \\Omega = \\frac{1}{S} \\sum_{\\lambda = 1}^{S} \\mathbf{T}  (S, \\lambda)

    Parameters
    ----------
    fg_matrix : np.ndarray
        The fine-grained matrix.
    initial_weights : np.ndarray or list
        Vector of initial weights.
    cg_map : list of lists
        List of all microstates in each macrostate.
    s : int
        Maximum trajectory length.
    time_horizon : int
        Time horizon.

    Returns
    -------
    occ : np.ndarray
        The occupancy matrix, computed as above.
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

    pass
