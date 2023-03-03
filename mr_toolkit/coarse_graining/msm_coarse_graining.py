"""Main module with code for coarse-graining transition matrices and computing bin-weights."""
import numpy as np
import scipy.linalg as la
import tqdm.auto as tqdm
try:
    import msmtools
except ModuleNotFoundError:
    print("msmtools not found...")
from copy import deepcopy


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

    # The boundary elements only have 1 neighbor, so give them an extra 1 from the missing neighbor.
    t_matrix[0,0] += 1
    t_matrix[-1,-1] += 1

    normalized_t_matrix = t_matrix / np.sum(t_matrix, axis=1)[:, np.newaxis]

    return normalized_t_matrix


def compute_avg_bin_weights(initial_weights, transition_matrix, max_s: int, lag: int = 1, min_s: int = 0,
                            leave=False):
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

    min_s : int
        Earliest trajectory point to use in sliding window calculation. Defaults to 0.

    leave : bool
        Leave TQDM progress bar on completion

    Returns
    -------
    wi_bar : np.ndarray (`n_states`)
        List of time-averaged weights for each bin
    """

    assert max_s >= lag, "Trajectory length S is shorter than lag!"

    weights = np.full_like(initial_weights, fill_value=0.0)

    # This needed a more efficient implementation...
    #       I think instead of raising the matrix to new powers, I can just keep multiplying the weights

    new_weights = initial_weights.copy()

    # This is to make the new method equivalent with the old
    weights += initial_weights

    for s in tqdm.tqdm(range(min_s, max_s - lag), desc="Sweeping trajectory length", leave=leave):

        new_weights = np.dot(
            new_weights, transition_matrix
        )
        weights += new_weights

    weights /= (max_s - lag + 1)

    # # Old implementation
    # weights = np.full_like(initial_weights, fill_value=0.0)
    # for s in range(min_s, max_s - lag + 1):
    #     new_weights = np.dot(
    #         initial_weights, np.linalg.matrix_power(transition_matrix, s)
    #     )
    #     weights += new_weights
    # weights /= (max_s - lag + 1)

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

    mat_pow = np.linalg.matrix_power(P, lag)

    # Iterate over every pair of n,m
    # TODO: There's probably a way to get rid of this explicit iteration
    #   At the very least, I could iterate over pairs of (m,n) but I'm not sure that's actually any faster than nested
    for m in range(num_cg_bins):
        for n in range(num_cg_bins):

            # For each of those pairs, iterate over each of the i, j elements

            T_dot = sum(np.dot(
                        w[cg_map[m]],
                        mat_pow[np.ix_(cg_map[m], cg_map[n])],
            ))

            # Old, explicitly looping implementation
            # T_iter = 0
            # for i in cg_map[m]:
            #     for j in cg_map[n]:
            #
            #         T_iter += w[i] * mat_pow[i, j]

            T[m,n] = T_dot

            # Finished an m,n pair, so normalize by the total weight of macrobin m
            microbins = cg_map[m]
            w_tot = np.sum(w[microbins])

            if normalize:
                if w_tot == 0:
                    T[m,n] = 0.0
                else:
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


def get_equil(transition_matrix: np.ndarray, normalize: bool = True, _round: int = 15) -> np.ndarray:
    """
    Computes the equilibrium distribution for an input transition matrix by taking the left-eigenvector of transition_matrix
    with an eigenvalue of 1.

    Parameters
    ----------
    transition_matrix : np.ndarray
        The transition matrix.

    _round : int, optional (12)
        Number of decimal places of precision to keep in the equil distribution.

    Returns
    -------
    equil : np.ndarray
        Equilibrium distribution for

    """

    evals, evecs = la.eig(transition_matrix, left=True, right=False)

    eval_1_idxs = np.where(np.isclose(evals, 1))[0]
    assert len(eval_1_idxs) > 0, 'No eigenvalues of 1 found!'
    # assert len(eval_1_idxs) < 2, 'Multiple eigenvalues of 1 found!'

    for i, eval_1_idx in enumerate(eval_1_idxs):
        # print(f"Checking eigval index {eval_1_idx}")

        # Check if distribution is positive semidefinite
        # eval_1_idx = eval_1_idxs[0]
        _equil_nonorm = evecs[:, eval_1_idx]
        _equil_nonorm[abs(_equil_nonorm) < 1e-15] = 0.0

        # Negative semidefinite will normalize to positive semidefinite
        if np.all(_equil_nonorm >= 0.0) or np.all(_equil_nonorm <= 0.0):
            # print(_equil_nonorm)
            break

        if i == len(eval_1_idxs)-1:
            raise Exception('No eigenvalue corresponding to a positive semidefinite range found')

    # print(evecs[eval_1_idxs])
    # print(evals[eval_1_idxs])
    # print(_equil_nonorm)

    normalization = [1, sum(_equil_nonorm)][normalize]
    _equil = _equil_nonorm / normalization

    # Sometimes there is a 0 imaginary component, ensure that this is indeed 0 and then if so, strip it.
    assert np.isreal(_equil).all()
    _equil = np.real(_equil)

    # Sanity check, no longer necessary
    msm_equil = msmtools.analysis.stationary_distribution(transition_matrix)
    try:
        assert np.all(np.isclose(_equil, msm_equil, rtol=.001)), f"Equil does not match msmtools equil. \n\t Eig: {_equil} "+\
                                                             f"\n\t MSMtools: {msm_equil}"
    except AssertionError as e:
        # log.raise(e)
        # print("Warning: Equil and MSMtools equilibrium dists don't agree")
        pass

    # Sanity check for stationarity, possibly more necessary than the previous
    first_step_equil = np.matmul(_equil, transition_matrix)
    equil_is_stationary = np.all(np.isclose(first_step_equil, _equil))
    assert equil_is_stationary, "Equilibrium not stationary under this 'stationary' solution!"

    # if not normalize:
    #     return _equil
    # else:
    #     return _equil, normalization
    return _equil
    # return msm_equil


def get_comm(transition_matrix: np.ndarray, statesA: list, statesB: list) -> np.ndarray:
    """
    Computes the committor for a given transition matrix, source states, and target states.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Transition matrix.
    statesA : array-like
        Source/origin state(s).
    statesB : array-like
        Target state(s).

    Returns
    -------
    committors : np.ndarray
        Array of committors to statesB for each bin.

    Raises
    ------
    AssertionError
        The solved committor distribution does not obey first-step stationarity.

    Todo
    -----
    Replace :code:`msmtools.analysis.committor` call with my own committor calculator.

    """

    raise DeprecationWarning

    _comms = msmtools.analysis.committor(transition_matrix, statesA, statesB)

    # _comms =

    # Sanity check for stationarity
    first_step_comms = np.matmul(transition_matrix, _comms)
    committor_is_stationary = np.all(np.isclose(first_step_comms, _comms))
    assert committor_is_stationary, "Committors not stationary under this solution!"

    return _comms


def get_hill_mfpt(ss_dist, T, target_mesostates):
    """
    Compute the MFPT via the Hill relation.

    From the Hill relation, the MFPT is the inverse flux into the target state, or

    .. math::  \\hillrelation


    Parameters
    ----------
    ss_dist: array-like
        Stationary distribution.

    T: array-like
        Transition matrix. (What BCs?)

    target_mesostates: array-like
        Target states for MFPT calculation.

    Returns
    -------
    MFPT: float
        First-passage time estimate.
    """

    # Get the flux into the target from each state
    flux = np.dot(ss_dist, T[:,target_mesostates])

    MFPT = 1./flux.sum()

    return MFPT

def get_naive_hill_mfpt(T_ss, ss_dist, target_mesostates, all_other_states):
    """
    This SHOULD just be an explicit, un-optimized version of get_hill_mfpt to make sure I got the linear algebra right
    """

    flux = 0


    for target in target_mesostates:
        for source in all_other_states:
            flux += ss_dist[source] * T_ss[source, target]

    return 1./flux


def make_ss(matrix, target_state, source_state, keep_sink_entry=False):
    """
    Given a matrix, add simple recycling boundary conditions from the target to the source.

    Parameters
    ----------

    matrix: array-like
        A transition matrix
    target_state: array-like
        Set of target states
    source_state: array-like
        Set of source states
    keep_sink_entry: boolean
        Keep one step in the target state if true, otherwise any entry into the target instead goes directly
        to the source

    Returns
    -------
    The matrix with recycling boundary conditions.
    """

    _ss_matrix = deepcopy(matrix)

    if not keep_sink_entry:
        _ss_matrix[:, source_state] += _ss_matrix[:, target_state]
        _ss_matrix[:, target_state] = 0.0
    else:
        _ss_matrix[target_state, :] = 0.0
        _ss_matrix[target_state, source_state] = 1.0

    return _ss_matrix
