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

    t_matrix = np.eye(num_bins, num_bins) * height_ratio + \
        np.eye(num_bins, num_bins, -1) + \
        np.eye(num_bins, num_bins,  1)

    normalized_t_matrix = t_matrix / np.sum(t_matrix, axis=1)[:, np.newaxis]

    return normalized_t_matrix

def coarse_grain(P: np.ndarray, cg_map: np.ndarray, w: np.ndarray):
    """
    Coarse-grains a fine-grained transition matrix according to some mapping of microstates to macrostates and weights
    over the microstates.

    Parameters
    ----------
    P : np.ndarray
        Fine-grained transition matrix.
    cg_map : list of lists
        List of all microstates in a macrostate.
    w : np.ndarray
        Microbin weights.

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

                    T[m,n] += w[i] * P[i,j]

            # Finished an m,n pair, so normalize by the total weight of macrobin m
            microbins = cg_map[m]
            w_tot = np.sum(w[microbins])
            T[m,n] /= w_tot

    return T


def compute_avg_bin_weights(initial_weights, transition_matrix, max_s):
    """
    Obtain the time-averaged bin weights for a lag of 1, described by

    .. math::  \\bar{w_i} = \\frac{1}{S} \\sum_{s=0}^{S-1} \\sum_k w_k(0) \\, (\\mathbf{P}^s)_{k \\rightarrow i}

    Parameters
    ----------
    initial_weights : np.ndarray or list
        List or array of initial microbin-weights.

    transition_matrix : np.ndarray
        Transition matrix.

    Returns
    -------
    wi_bar : np.ndarray
        List of time-averaged weights for each bin
    """


    weights = np.full_like(initial_weights, fill_value=0.0)
    n_bins = len(transition_matrix)

    for s in range(max_s):

        new_weights = np.dot(initial_weights, np.linalg.matrix_power(transition_matrix, s))
        weights += new_weights
        # for k in range(n_bins):
        #
        #     new_weights = initial_weights[k] * np.linalg.matrix_power(transition_matrix, s)[k,:]
        #     print(new_weights)
        #     weights += new_weights

    weights /= max_s

    return weights
