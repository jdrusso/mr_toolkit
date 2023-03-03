import logging
from rich.logging import RichHandler
import numpy as np
from copy import deepcopy
import tqdm.auto as tqdm
from deeptime.clustering import KMeansModel

from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgraph
from scipy.sparse.sputils import isdense

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger()
log.setLevel(logging.INFO)


def find_connected_sets(C, directed=True):
    """
    This implementation is taken from msmtools.estimation.sparse.connectivity, at commit 9312660.
    See the original at https://github.com/markovmodel/msmtools/blob/devel/msmtools/estimation/sparse/connectivity.py#L30
    Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        square matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.
       
    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in
        the corresponding connected component.
    """
    if isdense(C):
        C = csr_matrix(C)

    M = C.shape[0]
    """ Compute connected components of C. nc is the number of
    components, indices contain the component labels of the states
    """
    nc, indices = csgraph.connected_components(
        C, directed=directed, connection="strong"
    )

    states = np.arange(M)  # Discrete states

    """Order indices"""
    ind = np.argsort(indices)
    indices = indices[ind]

    """Order states"""
    states = states[ind]
    """ The state index tuple is now of the following form (states,
    indices)=([s_23, s_17,...,s_3, s_2, ...], [0, 0, ..., 1, 1, ...])
    """

    """Find number of states per component"""
    count = np.bincount(indices)

    """Cumulative sum of count gives start and end indices of
    components"""
    csum = np.zeros(len(count) + 1, dtype=int)
    csum[1:] = np.cumsum(count)

    """Generate list containing components, sort each component by
    increasing state label"""
    cc = []
    for i in range(nc):
        cc.append(np.sort(states[csum[i] : csum[i + 1]]))

    """Sort by size of component - largest component first"""
    cc = sorted(cc, key=lambda x: -len(x))

    return cc


def find_traps(transition_matrix):

    atol = 1e-8 # numpy default

    # Identify empty states (row and col sum to 0)
    empty_states = np.argwhere(np.isclose(np.sum(transition_matrix, axis=1), 0, atol=atol) &
                               np.isclose(np.sum(transition_matrix.T, axis=1), 0, atol=atol))

    # Identify strict sinks (Row sums to 0)
    sink_states = np.argwhere(np.isclose(np.sum(transition_matrix, axis=1), 0, atol=atol))

    # Identify strict sources (Column sums to 0)
    source_states = np.argwhere(np.isclose(np.sum(transition_matrix.T, axis=1), 0, atol=atol))

    # Identify disjoint states (Includes islands and self-transition-only (which are a special case of islands anyway ))
    connected_sets = find_connected_sets(transition_matrix)
    disjoint_states = []
    if len(connected_sets) > 1:
        disjoint_states = np.concatenate(connected_sets[1:]).flatten()

    from functools import reduce
    all_trap_states = reduce(np.union1d, (empty_states, sink_states, source_states, disjoint_states))

    return all_trap_states, (empty_states, sink_states, source_states, disjoint_states)


def clean_matrix(transition_matrix):

    cleaned_transition_matrix = transition_matrix.copy()
    all_trap_states, (empty_states, sink_states, source_states, disjoint_states) = find_traps(transition_matrix)

    while len(all_trap_states) > len(empty_states):
        print(f"{len(all_trap_states) - len(empty_states)} non-empty trap states exist")
        cleaned_transition_matrix[all_trap_states,:] = 0.0
        cleaned_transition_matrix[:,all_trap_states] = 0.0
        all_trap_states, (empty_states, sink_states, source_states, disjoint_states) = find_traps(
            cleaned_transition_matrix)

    return cleaned_transition_matrix, all_trap_states, (empty_states, sink_states, source_states, disjoint_states)



def transform_stratified(cluster_centers, projection, bin_boundaries, tic_rmsd):
    """
    Apply a set of stratified clusters to a TICA projection

    :param cluster_centers:
    :param projection:
    :param bin_boundaries:
    :param tic_rmsd:
    :return:
    """

    discretized = [np.full(len(x[1][0]), fill_value=np.nan) for x in projection]
    traj_starts = np.cumsum([len(x) for x in discretized])

    cluster_offset = 0

    for i, (bin_lower, bin_upper) in tqdm.tqdm(
            enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])),
            total=len(bin_boundaries)-1):

        # Make a KMeans estimator using only the cluster centers in this bin
        center_idxs_in_bin = np.argwhere(
            (cluster_centers[:, tic_rmsd] >= bin_lower) &
            (cluster_centers[:, tic_rmsd] < bin_upper)
        ).flatten()

        _clustering = KMeansModel(
            cluster_centers[center_idxs_in_bin],
            metric="euclidean",
            # fixed_seed=True
        )
        print(f"{center_idxs_in_bin.shape} centers in bin {i}")


        # For each trajectory, get its points that fall in the current bin
        all_traj_idxs_in_bin = []
        all_traj_points_in_bin = []
        for traj_idx, (traj_time_indices, traj_projection) in enumerate(projection):

            # print(f"Traj shape is {traj_projection.shape}")
            assert traj_projection.shape[0] == 10

            # Get the indices of points in the current bin
            traj_idxs_in_bin = np.argwhere(
                (traj_projection[tic_rmsd, :] >= bin_lower) &
                (traj_projection[tic_rmsd, :] < bin_upper)
            ).flatten()

            all_traj_idxs_in_bin.append(traj_idxs_in_bin)

            if len(traj_idxs_in_bin) > 0:

                assert len(_clustering._cluster_centers) > 0, f"Trajs but no cluster centers in bin {i}! " \
                                                              f"{bin_lower} <-> {bin_upper}" \
                                                              f"\n Traj points: {traj_idxs_in_bin}"

            # Now also get the actual points, to make clustering easier
            all_traj_points_in_bin.append(traj_projection[:, traj_idxs_in_bin])

        all_points_together = np.hstack(all_traj_points_in_bin).T
        # print(f"All together, shape is {all_points_together.shape}")

        if all_points_together.shape[0] == 0:
            print(f"No trajs in this set had points in bin {i}")
            continue

        # Now discretize the parts of each trajectory that are in this bin
        for traj_idx in range(len(projection)):

            traj_idxs_in_bin = all_traj_idxs_in_bin[traj_idx]
            traj_points_in_bin = all_traj_points_in_bin[traj_idx].T

            # print(f"{traj_idxs_in_bin.shape} points in bin {i}")

            transformed = _clustering.transform(traj_points_in_bin) + cluster_offset

            # print(transformed)

            discretized[traj_idx][traj_idxs_in_bin] = transformed

        cluster_offset += len(_clustering.cluster_centers)
        # print(f"Cluster offset is now {}")

    return discretized, cluster_centers, bin_boundaries


def remap_trajs(trajs):
    """
    Take a set of state trajectories, and remap the states to be consecutive. I.e., [0,1,3,5,6] becomes [0,1,2,3,4]

    Parameters
    -----------
    trajs: array-like
        Trajectories to remap

    Returns
    -------
    array-like
        Remapped trajectories, with all consecutive states.
    """

    trajs = deepcopy(trajs)
    # _trajs = np.array(trajs)
    # states = sorted(np.unique(_trajs))
    states = sorted(np.unique([i for s in trajs for i in s]))
    n_states = len(states)

    # Map each state to a consecutive index, so we can work with these moving forward
    consecutive_states = {state: i for i, state in enumerate(states)}

    # Convert each trajectory to consecutive states
    for i, traj in enumerate(trajs):
        trajs[i] = [consecutive_states[point] for point in traj]

    return trajs, n_states, consecutive_states


def optimized_resliced_reweighted(_trajs, n_iterations, _N, n_states,
                             lagtime=1,
                             _initial_weights=None, return_matrices=False,
                             last_frac=1.0,
                             # reweight_last_point=False,
                             debug=False,
                             min_weight=1e-12,
                             convergence=None
                             ):
    """
    Do Markov model building, using iterative reweighting.

    Parameters
    ----------
    _trajs : array-like
        Set of discretized trajectories
    n_iterations: int
        Number of iterations for iterative reweighting
    _N: int
        Fragment length for reslicing
    n_states: int
        Number of discrete states
    _initial_weights: array-like
        (Optional) Initial weights for reweighting. Defaults to uniform.
    return_matrices: bool
        (Optional, default False) Return intermediate weighted transition matrices
    last_frac: float
        (Optional, default 1.0) Only use transitions from the last_frac fraction of the fragments.
    debug: bool
        (Optional, default False) Enables some additional output.

    Returns
    -------
    array
        Steady-state distributions obtained from reweighting
    array (if return_matrices=True)
        Associated transition matrices
    """

    # if reweight_last_point:
    #     print("*** Reweighting on last point!")
    reweight_last_point = False

    assert _N > 0, "N must be > 0"

    # A lag time of 1 requires _N = 1 or greater
    # At a lag time of 2
    assert lagtime <= _N, "Lag time must be <= N (fragment length)"

    # ! Set up the initial distribution (uniform if not specified)
    # n_states = len(_initial_weights)

    # ! Compute initial weights
    _counts = np.zeros(shape=n_states)

    # These are counts of all valid fragment start points
    # A fragment can start at any point up to len-_N from the end
    # TODO: Because of this line, trajectories MUST be equal length and in an array.
    _states, _state_counts = np.unique(_trajs[:, :-_N], return_counts=True)
    _counts[_states] = _state_counts

    # If no initial weights were specified, then just weight all counts as 1
    if _initial_weights is None:
        # It'll get normalized later, and this keeps them to be pure counts
        _initial_weights = np.ones(n_states)
        transition_weights = _initial_weights

    # If you specified initial weights over the states, normalize by the number of fragment starts in each
    #   state.
    else:
        assert np.isclose(sum(_initial_weights), 1.0), "Transition weights not normalized"
        transition_weights = np.divide(_initial_weights, _counts,
                                       out=np.zeros_like(_initial_weights), where=_counts > 0)

    assert len(transition_weights) == n_states, "Wrong number of transition weights, doesn't match states"

    # ! Build fragment count matrices
    # weighted_count_matrix = np.zeros((n_states, n_states))
    # all_traj_count_matrix = np.zeros((n_states, n_states))
    _count_matrices = np.zeros(shape=(n_states, n_states, n_states))

    total_steps = len(_trajs) * len(_trajs[0][:-lagtime])
    with tqdm.tqdm(
        total=total_steps, desc="Building count matrices", miniters=int(total_steps/1000)
    ) as pbar:
        # TODO: Parallelize this loop over trajectories.
        #   Better than parallelizing over start points, less communication
        for _traj in _trajs:

            # * Build a transition matrix for transitions in fragments starting from each initial point
            # I.e., we have a count matrix for each initial state (count matrix from all fragments starting in
            #   state 0 for example)
            # Since initial states are weighted the same, we can apply the weights after building the count
            #   matrix and only have to build it once.

            # Only go up to len-1 here, because the last point isn't a transition and
            #   we're going over transitions, not fragments
            for transition_start_point in range(0, len(_traj)-lagtime):

                # Get all points that could be the start of a
                #   fragment containing this transition
                # Example: For 2-step fragments with 1 transition, _N=1, this range goes from
                #   transition_start - 1 + 1 to transition_start+1
                # This would give you a range of [transition_start]

                # The earliest fragment this can be a part of is one where it's the last transition start point, which
                #   was _N+1 ago
                # Make sure we don't go negative, though
                # Another way of putting this: The earliest fragment that can contain this transition must also
                #   contain the transition end point
                first_fragment_start = max(0, transition_start_point + lagtime - _N)

                # The last possible fragment start is _N back from the end of the trajectory, because fragments
                #   are fixed length. (i.e., you don't get a bunch of increasingly short fragments at the end)
                # 3/28 removed a +1 from transition_start_point...
                # latest_fragment_start = min(transition_start_point, len(_traj)-_N)
                latest_fragment_start = min(first_fragment_start+1, len(_traj)-_N)

                # For my last_n starts, I want to chop off latest_fragment_start. Before, it was bounded by the
                #   minimum of the transition start point (ensure it doesn't go before the start), and the length of
                #   the full trajectory I'm reslicing.
                # When last_frac = 1.0, this should reproduce the original behavior
                # i.e., first_fragment_start is transition_statr_point-_N+1, so min() becomes
                #   min(
                #      transition_start_point+1,
                #      (transition_start_point - _N + 1) + _N  = transition_start_point + 1
                #   )
                latest_fragment_start = min(latest_fragment_start,
                                            int(first_fragment_start + (_N * last_frac)))
                # latest_fragment_start = min(latest_fragment_start,
                #                             first_fragment_start + ())


                fragment_start_idxs = range(first_fragment_start, latest_fragment_start)

                # fragment_start_idxs = range(max(0, transition_start_point-_N+1), transition_start_point+1)

                # This gives me the start point of every fragment containing this transition.
                # TODO: To reweight by last point, just increment all of these start point indices by _N.
                if reweight_last_point:
                    try:
                        fragment_start_points = _traj[np.array(fragment_start_idxs) + _N]
                    except IndexError as e:
                        print(fragment_start_idxs)
                        print(_N)
                        raise e
                else:
                    fragment_start_points = _traj[fragment_start_idxs]

                transition_from = _traj[transition_start_point]
                transition_to = _traj[transition_start_point+lagtime]


                # If fragment_start_points is [3,3,1], then matrix 3 needs to get +2, and matrix 1 needs to get +1
                # _count_matrices[fragment_start_points, transition_from, transition_to] += 1
                fragment_states, fragment_counts = np.unique(fragment_start_points, return_counts=True)
                _count_matrices[fragment_states, transition_from, transition_to] += fragment_counts

                if debug:
                    d = {0:'A', 1:'B', 2:'C'}
                    print(f"Transition from {d[transition_from]} --> {d[transition_to]} counted for {list(zip(fragment_states+1, fragment_counts))}")

                pbar.update(1)

        # all_traj_count_matrix += _count_matrices

    # ! Iteratively: assign weights and compute stationary
    stationary_distributions = []
    matrices = []
    all_transition_weights = [transition_weights]
    prev_kl = 0
    with tqdm.tqdm(total=-np.log10(convergence), disable=convergence is None, desc="Reweighting convergence") as pbar:
        for _iter in range(n_iterations):

            # * Get the new weighted count matrix
            weighted_count_matrix = np.einsum('ijk,i->jk', _count_matrices, transition_weights)

            # * Row-normalize into a transition matrix

            # weighted_count_matrix, traps, (empty, _, _, _) = clean_matrix(weighted_count_matrix)
            row_sums = np.sum(weighted_count_matrix, axis=1)

            traps = []
            empty = []
            good_states = np.setdiff1d(np.arange(weighted_count_matrix.shape[0]), traps)

            try:
                transition_matrix = np.divide(
                    weighted_count_matrix.T,
                    row_sums,
                    out=np.zeros_like(weighted_count_matrix),
                    where=np.isin(np.arange(weighted_count_matrix.shape[0]), good_states) & (row_sums > 0),
                ).T
            except np.linalg.LinAlgError as e:

                print(_iter)

                raise e

            # Note - this is NOT a copy!
            matrices.append(transition_matrix)

            # * Get the stationary distribution
            evals, evecs = np.linalg.eig(transition_matrix.T)

            max_eig_index = np.argmin(1 - evals)

            stationary = np.real(evecs[:, max_eig_index]) / np.real(
                sum(evecs[:, max_eig_index])
            )

            # Clean any VERY small values that are below my minimum weight cutoff
            stationary[(stationary < 0) & (stationary > -min_weight)] = 0.0
            stationary = stationary / sum(stationary)

            # HACK: Sometimes you'll get a stationary distribution with everything in one state.. probably points to a
            #   deeper problem
            # if len(np.argwhere(stationary).flatten()) == 1:
            i = 0
            bad_stationary = False
            while len(np.argwhere(stationary).flatten()) == 1 or np.any(stationary/sum(stationary) < 0):
                i += 1

                if i >= len(evals):
                # if evals[i] < 0.9:
                    log.critical(f'No good stationary solution exists! Stopping iteration at iter {_iter}')
                    bad_stationary = True
                    break
                    # assert False, "No stationary solution could be found"

                if len(np.argwhere(stationary).flatten()) == 1:
                    log.warning(f"Stationary solution {i} is all in one bin in iter {_iter} -- picking next-biggest eigenvalue")
                if np.any(stationary/sum(stationary) < 0):
                    log.warning(f'Stationary solution {i} is not positive semidefinite, trying the next one')
                max_eig_index = np.argsort(1-evals)[i]
                stationary = np.real(evecs[:, max_eig_index]) / np.real(
                    sum(evecs[:, max_eig_index])
                )

            if bad_stationary:
                break

            # If any probabilities are zero that were not zero before, set them to the minimum weight and renormalize
            if _iter > 0:
                below_min = np.argwhere((stationary_distributions[-1] > 0) & (stationary < min_weight)).flatten()
                below_min = np.setdiff1d(below_min, traps)
                if len(below_min) > 0:
                    log.debug(f"In iter {_iter}, {len(below_min)} non-trap states with nonzero probabilities dropped "
                              f"below minimum weight in the new distribution..."
                              f" Setting them to {min_weight} and renormalizing.")
                    stationary[below_min] = min_weight
                    stationary = stationary / stationary.sum()

            assert np.isclose(stationary.sum(), 1.0), f"Stationary distribution not normalized in iter {_iter}!"
            assert np.all(stationary >= 0), \
                f"Stationary distribution not all positive!"

            stationary_distributions.append(stationary)

            if _iter > 1 and convergence is not None:
                # rms_change = np.sqrt(np.mean(np.power(stationary - stationary_distributions[-1], 2)))
                # print(f"RMS change was {rms_change:.1e}")

                # This is the CHANGE in "energy"
                abs_kl_unweighted = np.nansum(np.abs(
                    np.log(stationary_distributions[-1]) - np.log(stationary_distributions[-2])
                ))
                # print(f"Abs unwgt. KL change was {abs_kl_unweighted:.5e}")

                pbar_increment = (-np.log10(abs_kl_unweighted)) - pbar.n
                pbar_increment = min(pbar_increment, convergence - pbar.n)
                pbar.update()


                if abs_kl_unweighted < convergence:
                    pbar.write(f"reweighted iteration is converged at iter {_iter}")
                    break

                # pbar.update((-np.log10(abs_kl_unweighted)) - pbar.n)


            # * Compute the new fragment weights
            # But I need to normalize transition weights by counts here!
            # transition_weights = stationary

            # This new stationary distribution is the
            new_stationary = stationary.copy()
            new_stationary[_counts == 0] = 0.0
            # new_stationary /= sum(new_stationary)
            new_stationary = np.divide(new_stationary, sum(new_stationary),
                                           out=np.zeros_like(new_stationary), where=new_stationary > 0)
            new_stationary = new_stationary/new_stationary.sum()

            assert np.isclose(new_stationary.sum(), 1.0), \
                f"New distribution not normalized in iter {_iter}! Sums to {new_stationary.sum()}"
            assert np.all(new_stationary >= 0), \
                f"New distribution not all positive!"

            transition_weights = np.divide(new_stationary, _counts,
                                           out=np.zeros_like(new_stationary), where=_counts > 0)


            all_transition_weights.append(transition_weights)


    # ! Return the final stationary distribution
    if return_matrices:
        return stationary_distributions, matrices, _count_matrices, _iter
    else:
        return stationary_distributions


# @profile
def build_resliced_msm(_trajs, _initial_weights, N, total_pbar=None):
    """
    Given a set of (consecutive-state) trajectories, build an MSM by reslicing.

    Parameters
    ----------
    _trajs : array-like
        A set of trajectories where the states are consecutive
    _initial_weights : array-like
        Initial weights for the states
    N : integer
        Fragment length. N=1 means 2-step (1 transition) trajectories.

    Returns
    -------
    array
        A transition matrix
    """
    assert N > 0, "N must be > 0"

    n_states = len(_initial_weights)
    count_matrix = np.zeros((n_states, n_states))

    # Go through each trajectory in the set
    # TODO: (Parallelize over this)
    # TODO: Maybe JIT compile this? idk

    _counts = np.zeros(shape=n_states)
    _states, _state_counts = np.unique(_trajs[:, :N], return_counts=True)
    _counts[_states] = _state_counts

    for _traj in _trajs:

        # I think we want to choose just 1 N, but might as well leave it flexible to multiple.
        # for N in range(N)
        for _N in [N]:


            # If you have a trajectory of length M, then the last start point will be M-_N
            # I.e., a trajectory of length 5 with _N=3 [0,1,2,3,4]
            #   The last start point will be [1], so len(trajectory) - _N - 1
            #   Because I want to slice up to and including the last start point, I drop the -1 in the slice below.
            #   For the above trajectory, that would be [0,1,2,3,4][:-3] = [0,1]
            # _states, _state_counts = np.unique(_traj[:-_N], return_counts=True)
            # _counts[_states] = _state_counts

            transition_weights = np.divide(_initial_weights, _counts,
                                           out=np.zeros_like(_initial_weights), where=_counts > 0)

            # * New implementation
            # ! Build a transition matrix for transitions in fragments starting from each initial point
            _count_matrices = np.zeros(shape=(n_states, n_states, n_states))
            # for transition_start_point in range(0, len(_traj)-_N):
            # Only go up to len-1 here, because the last point isn't a transition
            # Here we're going over transitions, not fragments
            # for transition_start_point in range(0, len(_traj)-1):
            for transition_start_point in range(0, len(_traj)-1):

                # Get the indices of all points that could be the start of a
                #   fragment containing this transition
                # For 2-step fragments with 1 transition, _N=1, and this range goes from
                #   transition_start - 1 + 1 to transition_start+1
                # This would give you a range of [transition_start]
                fragment_start_idxs = range(max(0, transition_start_point-_N+1), transition_start_point+1)
                fragment_start_points = _traj[fragment_start_idxs]

                transition_from = _traj[transition_start_point]
                transition_to = _traj[transition_start_point+1]

                # This line is wrong -- it only increments each matrix once
                # I.e., if fragment_start_points is [3,3,1], then both matrices 1 and 3 will only get +1.
                # Really, matrix 3 needs to get +2, and matrix 1 needs to get +1
                # _count_matrices[fragment_start_points, transition_from, transition_to] += 1
                fragment_states, fragment_counts = np.unique(fragment_start_points, return_counts=True)
                _count_matrices[fragment_states, transition_from, transition_to] += fragment_counts

                total_pbar.update(1)

            # Finish by taking the weighted sum of these matrices and normalizing
            # "Markov-weighted Markov matrix" :^)
            total_count_matrix = np.einsum('ijk,i->jk', _count_matrices, transition_weights)
            count_matrix += total_count_matrix

    return count_matrix

            # * Old implementation
            # # For each trajectory, we want to split out every fragment of length N
            # # Sanity check on indexing: If I have a 5-step trajectory, with N=3, then my valid start points are
            # #   0, 1 (since N=3 means 4-step trajectories)
            # # Then, my fragments range from 0 .. 3 and 1 .. 4
            # # Therefore, my start points need to be _N+1 in my list slicing, so that it includes the _N point.
            # for start_point in range(0, len(_traj)-_N):
            #
            #     fragment = _traj[start_point:start_point+_N+1]
            #     fragment_initial_weight = _initial_weights[fragment[0]]
            #     transition_weight = fragment_initial_weight / _counts[fragment[0]]
            #
            #
            #     # This is a pretty cool way to iterate over these transitions that I never thought about before
            #     # Neither of these seems particularly faster, they're basically dead even (impressive, python)
            #     # TODO: Avoid explicitly looping over this?
            #     # TODO: or at least parallelize it...
            #     for (i, j) in zip(fragment[:-1], fragment[1:]):
            #     # for _idx in range(len(fragment)-1):
            #
            #         # Increment the matrix using the initial weight OF THE FRAGMENT
            #         count_matrix[i, j] += transition_weight
            #         # count_matrix[fragment[_idx], fragment[_idx+1]] += transition_weight
            #
            #         # It feels like there's a much faster algorithm for this, but I need to draw it out
            #         # The trick would be not iterating over fragments, and instead always looking forward X amount?
            #
            #     # Counts n_reweighted * n_trajs * (len(traj) - N)
            #     total_pbar.update(1)

    return count_matrix


# @profile
def build_msm(trajs, initial_weights=None, reslicing=False, N=None, total_pbar=None, normalize_initial=True):
    """
    Build a Markov model from a set of trajectories, and a set of weights for their initial points.
    """

    _trajs, n_states, consecutive_states = remap_trajs(trajs)
    _trajs = np.array(_trajs)

    # Use uniform initial weights if none are provided
    if initial_weights is None:
        log.debug("No initial weights supplied, using uniform")
        initial_weights = np.ones(shape=n_states) / n_states

    assert len(initial_weights) == n_states, "Initial states has wrong dimensionality"
    assert np.isclose(sum(initial_weights), 1), "Initial weights not normalized"

    # Maybe janky to put this logic in here but w/e
    if reslicing:
        assert N is not None, "Must provide an N for reslicing"
        resliced_transition_matrix = build_resliced_msm(_trajs, initial_weights, N, total_pbar=total_pbar)
        return resliced_transition_matrix, consecutive_states, None, None
        # return transition_matrix, consecutive_states, total_count_matrix, trajectory_weights


    ## Get trajectory weights
    trajectory_weights = np.zeros(shape=n_states)
    initial_points = [traj[0] for traj in _trajs]
    values, counts = np.unique(initial_points, return_counts=True)
    #     print(values, counts)
    initial_point_counts = {value: count for value, count in zip(values, counts)}

    # Now, divide the trajectory weights by the number of trajectories
    # Explicitly zero out anything w/ no initial points and redistribute its weight, though if I didn't, this
    #    would be handled in normalization elsewhere
    lost = 0

    # This handles the saeme thing as the 'not in' in the for loop below... Probably don't need it in both places
    empty_states = np.setdiff1d(range(n_states), values)
    #     print(empty_states)
    initial_weights[empty_states] = 0.0
    # if normalize_initial:
    initial_weights /= sum(initial_weights)

    for state in range(n_states):

        # I don't think this even runs, the setdiff1d above handles this already.
        # If this state isn't in the initial points of the trajectory, we can't reweight by it
        # So, set it to 0.0, and move on.
        # I don't think I need to do anything special here w.r.t. normalization, the nonzero
        #     points will still be reweighted relative to each other (there will just be an overall)
        #     normalization.
        if state not in values:
            trajectory_weights[state] = 0.0
            lost += initial_weights[state]
            continue

        trajs_starting_in_state = initial_point_counts[state]

        # Split up state weight over all the trajectories starting in that state
        trajectory_weights[state] = (
            initial_weights[state] / initial_point_counts[state]
        )

    n_traj_in_state = np.zeros(n_states)
    n_traj_in_state[values] = counts

    # if normalize_initial:
    try:
        assert np.isclose(sum(trajectory_weights * n_traj_in_state), 1.0)
    except AssertionError as e:
        raise e

    # For reweighted iteration, do I want to:
    #    [Did this] - - - Weight the count matrices by the trajectory weights, and then add them and norm
    #    - Weight the transition matrices by the trajectory weights, and then add them
    # Looks like these are equivalent, I'll do the former because it makes the normalization easier
    total_count_matrix = np.zeros((n_states, n_states))
    true_count_matrix = np.zeros((n_states, n_states))

    _total = sum([len(traj[1:]) for traj in _trajs])
    with tqdm.tqdm(
        total=_total, desc="MSM Building", miniters=int(_total/1000)
    ) as pbar:
        for traj in _trajs:

            traj_weight = trajectory_weights[traj[0]]
            traj_count_matrix = np.zeros_like(total_count_matrix)

            for idx in range(1, len(traj)):

                _from = traj[idx - 1]
                _to = traj[idx]

                traj_count_matrix[_from, _to] += 1
                pbar.update(1)

            true_count_matrix = true_count_matrix + traj_count_matrix

            # TODO: ..wut? what is this? why was I weighting these, versus just taking the plain count matrix?
            #   And why did this give an answer consistent with PyEmma for the other data? Was it just close enough?
            total_count_matrix = total_count_matrix + (traj_count_matrix * traj_weight)

    row_sums = np.sum(true_count_matrix, axis=1)

    # transition_matrix = (total_count_matrix.T / row_sums).T
    transition_matrix = np.divide(
        true_count_matrix.T,
        row_sums,
        out=np.zeros_like(true_count_matrix),
        where=row_sums != 0,
    ).T
    transition_matrix[row_sums == 0] = 0.0

    return transition_matrix, consecutive_states, true_count_matrix, trajectory_weights


# @profile
def do_reweighted(trajs, n_iters, reslicing=False, initial_pSS=None, N=None, _pbar=None, _total_pbar=None):
    """
    Estimate equilibrium distributions using reweighted distributions

    :param trajs: array-like, trajectories to build MSM from
    :param n_iters: int, number of iterations
    :param reslicing: bool, whether to weight trajectories (False), or each point (True)
    :param initial_pSS: array-like, Initial distribution for iteration 0. None will use a uniform initial.
    :return: Array of distributions after each iteration, array of active states in each iteration
    """
    prev_pSS = initial_pSS
    result_distributions = []
    states = []

    for i in range(n_iters):

        # _total_pbar

        tmat, state_map, cmat, weights = build_msm(
            trajs, initial_weights=prev_pSS, reslicing=reslicing, N=N, total_pbar=_total_pbar
        )

        # Normalize
        row_sums = np.sum(tmat, axis=1)
        tmat = np.divide(
            tmat.T,
            row_sums,
            out=np.zeros_like(tmat),
            where=row_sums != 0,
        ).T
        tmat[row_sums == 0] = 0.0

        evals, evecs = np.linalg.eig(tmat.T)

        # TODO: Why is this necessary? Some transition matrices have 0-rows, I think that's the reason.
        max_eig_index = np.argmin(1 - evals)
        stationary = np.real(evecs[:, max_eig_index]) / np.real(
            sum(evecs[:, max_eig_index])
        )

        prev_pSS = stationary

        result_distributions.append(np.array(stationary))
        states.append(list(state_map.keys()))

        if _pbar is not None:
            _pbar.update(1)

    return np.array(result_distributions), states


def split_trajectories(original_single_trajectories, fragment_len):
    new_single_trajs = []

    for og_single_traj in original_single_trajectories:

        traj_len = len(og_single_traj)
        n_splits = traj_len // fragment_len
        for i in range(n_splits + 1):
            fragment = og_single_traj[
                       fragment_len * i: min(traj_len, fragment_len * (i + 1))
                       ]
            if len(fragment) < 2:
                continue
            new_single_trajs.append(fragment)
        pass

    return new_single_trajs


def split_data(filename, n_trajs):
    data_file = filename
    label = filename.split("_")[0].split('/')[1]

    state_traj = [int(x) for x in np.loadtxt(data_file)]
    traj_len = len(state_traj) // n_trajs

    return [
        [
            state_traj[(start) * traj_len : (start + 1) * traj_len]
            for start in range(n_trajs)
        ],
        label,
    ]
