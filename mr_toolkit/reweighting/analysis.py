"""
Code for reweighting MSMs and obtaining reweighted estimates of steady-state.
"""
import deeptime.markov.tools.analysis
import numpy as np
from scipy.special import rel_entr
import pandas as pd
import mr_toolkit.trajectory_analysis.traj_analysis as ta
import logging
import pyemma
from .splicing import get_receiving_distribution, splice_trajectories, iterative_trajectory_splicing

log = logging.getLogger()

try:
    from msm_we.fpt import MarkovFPT
except ImportError:
    log.warning("msm_we not found, fpt_distribution calculations will be unavailable")


def compute_reweighted_stationary(
        discrete_trajectories,
        N,
        lag,
        n_clusters,
        last_frac=1.0,
        min_weight=1e-12,
        n_reweighting_iters=100,
):
    """
    Estimates a stationary distribution from a discrete trajectory using reweighted MSMs.

    Parameters
    ----------
    discrete_trajectories: array-like
        2-D array or list of lists with discretized trajectories
    N: int
        Fragment length for reweighting
    lag: int
        Lagtime used in reweighting MSMs
    n_clusters: int
        Number of total states in the reweighted models (or in the discretization)
    last_frac: float
        Fraction of the trajectories to use. I.e., last_frac=0.25 only uses the last 1/4 of the trajectories.
    min_weight: float
        Minimum bound on weights during reweighting iteration
    n_reweighting_iters: int
        Maximum number of reweighting iterations

    Returns
    -------
    (Set of state indices, Stationary distributions at each reweighting iteration, Total number of iterations before
        convergence, estimated transition matrices at each reweighting iteration)
    """

    reweighted_stationaries = np.empty(shape=(n_reweighting_iters, n_clusters))
    reweighted_matrices = np.empty(
        shape=(
            n_reweighting_iters,
            n_clusters,
            n_clusters,
        )
    )

    try:
        (
            reweighted_distributions,
            _reweighted_matrices,
            weighted_count_matrices,
            last_iter,
        ) = ta.optimized_resliced_reweighted(
            discrete_trajectories,
            n_reweighting_iters,
            N,
            lagtime=lag,
            n_states=n_clusters,
            last_frac=last_frac,
            return_matrices=True,
            min_weight=min_weight,
            convergence=1e-10,  # Convergence threshold in units of kT
        )

    except (AssertionError, np.linalg.LinAlgError) as e:
        # This may trip if something goes awry in solving the matrices

        log.error(e)
        for i in range(n_reweighting_iters):
            reweighted_stationaries[i, :] = np.nan
            reweighted_matrices[i, :, :] = np.nan

        last_iter = 0

    else:
        for i, _distribution in enumerate(reweighted_distributions):
            reweighted_stationaries[i, :] = _distribution
            reweighted_matrices[i] = _reweighted_matrices[i]

    states = np.arange(n_clusters)
    return states, reweighted_stationaries, last_iter, reweighted_matrices


def get_set_kls(distributions):
    """
    Get KL divergences between multiple sets of distributions.

    I.e., a (4x10) input corresponds to 4x 10-element distributions.
    This would return an upper triangular 4x4 matrix with the unique pairwise KL-divergences.

    Parameters
    ----------
    distributions

    Returns
    -------

    """

    n_sets = len(distributions)

    kls = np.full(
        shape=(n_sets, n_sets),
        fill_value=np.nan,
    )
    for x, y in np.array(np.triu_indices(n_sets, 1)).T:
        setA_distribution = distributions[x]
        setB_distribution = distributions[y]

        kl_sum = get_kl(setA_distribution, setB_distribution, return_nan=True)
        kls[x, y] = kl_sum

    mean = np.nanmean(kls)

    return mean


def get_kl(test_dist, ref_dist, return_nan=False):
    """
    Obtain the KL divergence between two distributions.

    :param test_dist: The distribution to test
    :param ref_dist: The reference distribution
    :param return_nan: If the KL divergence is invalid for some reason, return NaN if true or -1 otherwise.
    :return: The KL divergence of the two distributions. If invalid, -1 or NaN depending on the value of return_nan.
    """
    elem_kl = rel_entr(test_dist, ref_dist, )
    kl_sum = np.nansum(elem_kl[elem_kl < np.inf])

    if kl_sum == 0:
        return [-1, np.nan][return_nan]

    return kl_sum


class AnalysisRun:
    """
    WARNING: You should almost certainly not use this!

    Convenience class to handle computing various observables from a trajectory set and storing relevant hyperparameters.
    """

    def __init__(
            self,
            _run,
            _reference,
            _trajectory_sets,
            dt=1,
            lag=None,
            # mfpt_method=metaparameters['mfpt_method'],
            metaparameters={},
    ):

        self.run = _run
        self.reference = _reference
        self.trajectory_sets = _trajectory_sets
        self.dt = dt
        self.lag = lag
        self.current_direction = None

        methods = [
            "histogram",
            "pyemma_rev",
            "pyemma_irrev",
            "naive",
            "reweighted",
            "resliced",
        ]
        multi_index = pd.MultiIndex.from_product(
            [range(metaparameters["n_trajectory_sets"]), methods],
            names=["trajectory set", "method"],
        )

        self.n_stratified_clusters = len(_reference)
        self.equil_df = (
            pd.DataFrame(
                index=multi_index,
                columns=range(self.n_stratified_clusters),
                dtype=np.float64,
            )
            .apply(pd.to_numeric)
            .sort_index()
        )

        # TODO: Generalize to more than 2 directions
        self.directions = ["unfolding", "folding"]
        multi_index = pd.MultiIndex.from_product(
            [range(metaparameters["n_trajectory_sets"]), self.directions, methods],
            names=["trajectory set", "direction", "method"],
        )
        self.ness_df = (
            pd.DataFrame(
                index=multi_index,
                columns=range(self.n_stratified_clusters),
                dtype=np.float64,
            )
            .apply(pd.to_numeric)
            .sort_index()
        )

        self.current_traj_set = -1
        self.active_df = None
        self.spliced_trajs = None
        self.current_direction = None

        self.metaparameters = metaparameters
        self.mfpt_method = self.metaparameters["mfpt_method"]

    def compute_avg_kl(self, method):

        # Compute average set-set KL
        kls = np.full(
            shape=(
                self.metaparameters["n_trajectory_sets"],
                self.metaparameters["n_trajectory_sets"],
            ),
            fill_value=np.nan,
        )

        for x, y in np.array(
                np.triu_indices(self.metaparameters["n_trajectory_sets"], 1)
        ).T:
            setA_converged_iter = self.equil_df.loc[(x, method)].values.shape[0]
            setB_converged_iter = self.equil_df.loc[(y, method)].values.shape[0]

            setA_reweighted_last = self.equil_df.loc[(x, method)].values.astype(float)
            setB_reweighted_last = self.equil_df.loc[(y, method)].values.astype(float)
            kl_sum = get_kl(setA_reweighted_last, setB_reweighted_last, return_nan=True)
            kls[x, y] = kl_sum

        mean_kl = np.nanmean(kls)
        std_kl = 2 * np.nanstd(kls) / np.sqrt(self.metaparameters["n_trajectory_sets"])

        return mean_kl, std_kl

    def compute_stationary(self, method, **kwargs):

        # TODO: Optionally disable logging / data storage here, so I can reuse this in compute_mfpt

        set_idx = self.current_traj_set
        assert self.active_df is not None, "No dataframe is active for storing results"

        stationary = np.zeros(shape=self.n_stratified_clusters)

        if method == "reweighted":
            (
                states,
                stationaries,
                last_iter,
                reweighted_matrices,
            ) = self.compute_reweighted_stationary(
                self.trajectory_sets[set_idx], **kwargs
            )

            assert np.isclose(
                np.sum(stationaries[:last_iter], axis=1), 1.0
            ).all(), "Stationary distributions not normalized!"

            # Compute KL to reference
            resliced_kl = get_kl(stationaries[0], self.reference)
            reweighted_kl = get_kl(stationaries[last_iter], self.reference)

            self.run.log_metric(f"set{set_idx}_resliced_kl", resliced_kl)
            self.run.log_metric(f"set{set_idx}_reweighted_kl", reweighted_kl)

            self.equil_df.loc[(set_idx, "resliced"), states] = stationaries[0]
            self.equil_df.loc[(set_idx, "reweighted"), states] = stationaries[last_iter]

            return stationaries[last_iter], reweighted_kl

        if method == "naive":
            _states, _stationary, tmatrix = self.compute_stationary_naive(
                self.trajectory_sets[set_idx]
            )
        elif method == "pyemma_rev":
            kwargs.pop('N')
            _states, _stationary, tmatrix = self.compute_pyemma_stationary(
                self.trajectory_sets[set_idx], reversible=True, **kwargs
            )
        elif method == "pyemma_irrev":
            kwargs.pop('N')
            _states, _stationary, tmatrix = self.compute_pyemma_stationary(
                self.trajectory_sets[set_idx], reversible=False, **kwargs
            )
        elif method == "histogram":
            _states, counts = np.unique(
                self.trajectory_sets[set_idx], return_counts=True
            )
            _stationary = counts / sum(counts)
        else:
            raise NotImplementedError("Invalid method specified")

        stationary[_states] = _stationary
        assert np.isclose(
            np.sum(stationary), 1.0
        ), "Stationary distribution not normalized!"

        # Compute KL to reference
        kl = get_kl(stationary, self.reference)

        self.run.log_metric(f"set{set_idx}_{method}_kl", kl)

        self.equil_df.loc[(set_idx, method), :] = stationary

        return stationary, kl

    @staticmethod
    def compute_stationary_naive(discrete_trajectories):

        tmatrix, state_map, cmatrix, weights = ta.build_msm(
            discrete_trajectories, reslicing=False, normalize_initial=False
        )
        evals, evecs = np.linalg.eig(tmatrix.T)
        max_eig_index = np.argmin(1 - evals)
        stationary = np.real(evecs[:, max_eig_index]) / np.real(
            sum(evecs[:, max_eig_index])
        )

        states = list(state_map.keys())

        return states, stationary, tmatrix

    @staticmethod
    def compute_pyemma_stationary(discrete_trajectories, lag, reversible=False):
        # TODO: Support different lagtimes

        pyemma_msm = pyemma.msm.estimate_markov_model(
            [x for x in discrete_trajectories], lag=lag, reversible=reversible
        )
        pyemma_stationary = pyemma_msm.stationary_distribution
        pyemma_states = pyemma_msm.active_set

        return pyemma_states, pyemma_stationary, pyemma_msm.transition_matrix

    def compute_reweighted_stationary(
            self,
            discrete_trajectories,
            N,
            lag,
            last_frac=None,
            min_weight=None,
            n_reweighting_iters=None,
            store_matrices=False,
    ):

        # TODO: This is obsolete now, remove this and just wrap a call to the static version

        if last_frac is None:
            last_frac = self.metaparameters.get("last_frac")
        if min_weight is None:
            min_weight = self.metaparameters.get("min_weight")
        if n_reweighting_iters is None:
            n_reweighting_iters = self.metaparameters.get("n_reweighting_iters")

        reweighted_stationaries = np.empty(shape=(n_reweighting_iters, self.n_stratified_clusters))
        reweighted_matrices = np.empty(
            shape=(
                n_reweighting_iters,
                self.n_stratified_clusters,
                self.n_stratified_clusters,
            )
        )

        try:
            (
                reweighted_distributions,
                _reweighted_matrices,
                weighted_count_matrices,
                last_iter,
            ) = ta.optimized_resliced_reweighted(
                discrete_trajectories,
                n_reweighting_iters,
                N,
                lagtime=lag,
                n_states=self.n_stratified_clusters,
                last_frac=last_frac,
                return_matrices=True,
                min_weight=min_weight,
                convergence=1e-10,  # Convergence threshold in units of kT
            )

        except (AssertionError, np.linalg.LinAlgError) as e:
            # This may trip if something goes awry in solving the matrices

            log.error(e)
            for i in range(n_reweighting_iters):
                reweighted_stationaries[i, :] = np.nan
                reweighted_matrices[i, :, :] = np.nan

            last_iter = 0

        else:
            for i, _distribution in enumerate(reweighted_distributions):
                reweighted_stationaries[i, :] = _distribution
                reweighted_matrices[i] = _reweighted_matrices[i]

        if store_matrices:
            index = pd.MultiIndex.from_product(
                [
                    range(last_iter + 1),
                    range(self.n_stratified_clusters),
                    range(self.n_stratified_clusters),
                ],
                names=["Iteration", "From", "To"],
            )
            reweighted_matrix_df = pd.DataFrame(
                np.array(reweighted_matrices).flatten(), index=index
            )
            reweighted_matrix_df.to_pickle(
                f"../results/{self.run.id}_set{self.current_traj_set}_reweighted_matrix_df.pkl"
            )

            index = pd.MultiIndex.from_product(
                [
                    range(self.n_stratified_clusters),
                    range(self.n_stratified_clusters),
                    range(self.n_stratified_clusters),
                ],
                names=["FragStart", "From", "To"],
            )
            count_matrix_df = pd.DataFrame(
                np.array(weighted_count_matrices).flatten(), index=index
            )
            count_matrix_df.to_pickle(
                f"../results/{self.run.id}_set{self.current_traj_set}_weighted_count_matrix_df.pkl"
            )

        states = np.arange(self.n_stratified_clusters)
        return states, reweighted_stationaries, last_iter, reweighted_matrices

    def iterative_trajectory_splicing(self,
                                      source_states,
                                      sink_states,
                                      splice_msm_lag=None,
                                      msm_reversible=False,
                                      target_steps_to_keep=1,
                                      convergence=1e-9,
                                      max_iterations=100):

        # This just wraps the external call for backwards-compatibility with some existing analysis scripts.

        if splice_msm_lag is None:
            print(f"No lag provided for splice MSM -- using set value of {self.lag}")
            splice_msm_lag = self.lag

        spliced_trajectories = iterative_trajectory_splicing(
            source_states=source_states,
            sink_states=sink_states,
            splice_msm_lag=splice_msm_lag,
            msm_reversible=msm_reversible,
            target_steps_to_keep=target_steps_to_keep,
            convergence=convergence,
            max_iterations=max_iterations,
            n_clusters=self.n_stratified_clusters
        )

        self.spliced_trajs = spliced_trajectories

    def splice_trajectories(
            self,
            source_states,
            sink_states,
            msm_lag=1,
            msm_reversible=False,
            target_steps_to_keep=1,
            trajs_to_splice=None,
            pbar_visible=True
    ):
        # This just wraps the external call for backwards-compatibility with some existing analysis scripts.

        set_idx = self.current_traj_set

        # Build an MSM to approximate the equilibrium distribution over the boundary states
        # TODO: Do we want to just use the PyEmma MSM? Or is there a better choice?
        if trajs_to_splice is None:
            trajs_to_splice = self.trajectory_sets[set_idx]

        spliced_trajs = splice_trajectories(
            trajs_to_splice=trajs_to_splice,
            source_states=source_states,
            sink_states=sink_states,
            msm_lag=msm_lag,
            msm_reversible=msm_reversible,
            target_steps_to_keep=target_steps_to_keep,
            pbar_visible=pbar_visible
        )

        self.spliced_trajs = spliced_trajs

    def compute_mfpt(self, method, source_states, target_states, **kwargs):
        set_idx = self.current_traj_set
        assert self.active_df is not None, "No dataframe is active for storing results"

        assert (
                self.current_direction is not None
        ), "No direction is specified for NESS/MFPTs"
        direction_index = self.directions.index(self.current_direction)

        if 'lag' in kwargs.keys():
            kwargs.pop('lag')
            print("Warning: Lag specification in arguments to compute_mfpt is unnecessary and unused")

        # Get the stationary distribution and transition matrix
        if method == "reweighted":

            (
                states,
                stationaries,
                last_iter,
                tmatrices,
            ) = self.compute_reweighted_stationary(self.spliced_trajs, lag=self.lag, **kwargs)

            if last_iter == 0:
                log.warning("Bad stationary distribution -- not attempting an MFPT")
                return None, None, None

            assert np.isclose(
                np.sum(stationaries[:last_iter], axis=1), 1.0
            ).all(), "Stationary distributions not normalized!"

            resliced_stationary = np.zeros(shape=self.n_stratified_clusters)
            reweighted_stationary = np.zeros(shape=self.n_stratified_clusters)
            resliced_tmatrix = np.zeros(
                shape=(self.n_stratified_clusters, self.n_stratified_clusters)
            )
            reweighted_tmatrix = np.zeros(
                shape=(self.n_stratified_clusters, self.n_stratified_clusters)
            )

            _resliced_stationary, _resliced_tmatrix = stationaries[0], tmatrices[0]
            resliced_stationary[states] = _resliced_stationary
            resliced_tmatrix[np.ix_(states, states)] = _resliced_tmatrix

            # resliced_mfpt = cg.get_hill_mfpt(resliced_stationary, resliced_tmatrix, target_states) * self.dt
            # resliced_mfpt = pyemma.msm.markov_model(resliced_tmatrix).mfpt(A=source_states, B=target_states)
            resliced_mfpt = self.mfpt(
                self.mfpt_method,
                resliced_tmatrix,
                source_states,
                target_states,
                lag=self.lag,
                dt=self.dt,
                stationary=resliced_stationary,
            )

            self.run.log_metric(
                f"set{set_idx}_resliced_mfpt_{self.current_direction}", resliced_mfpt
            )
            self.ness_df.loc[
                (set_idx, self.current_direction, "resliced"), states
            ] = resliced_stationary

            _reweighted_stationary, _reweighted_tmatrix = (
                stationaries[last_iter],
                tmatrices[last_iter],
            )
            reweighted_stationary[states] = _reweighted_stationary
            reweighted_tmatrix[np.ix_(states, states)] = _reweighted_tmatrix

            # reweighted_mfpt = cg.get_hill_mfpt(reweighted_stationary, reweighted_tmatrix, target_states) * self.dt
            # reweighted_mfpt = pyemma.msm.markov_model(reweighted_tmatrix).mfpt(A=source_states, B=target_states)
            reweighted_mfpt = self.mfpt(
                self.mfpt_method,
                reweighted_tmatrix,
                source_states,
                target_states,
                lag=self.lag,
                dt=self.dt,
                stationary=reweighted_stationary,
            )

            if not np.isnan(reweighted_mfpt):
                self.run.log_metric(
                    f"set{set_idx}_reweighted_mfpt_{self.current_direction}",
                    reweighted_mfpt,
                )

            self.ness_df.loc[
                (set_idx, self.current_direction, "reweighted")
            ] = reweighted_stationary
            return (
                (resliced_mfpt, reweighted_mfpt),
                (resliced_tmatrix, reweighted_tmatrix),
                (resliced_stationary, reweighted_stationary),
            )

        if method == "naive":
            print("Warning: Naive estimator assumes a lag of 1!")
            _states, _stationary, _tmatrix = self.compute_stationary_naive(
                self.spliced_trajs
            )
        elif method == "pyemma_rev":
            if 'N' in kwargs.keys(): kwargs.pop('N')
            _states, _stationary, _tmatrix = self.compute_pyemma_stationary(
                self.spliced_trajs, reversible=True, lag=self.lag  # TODO: Manage lag correctly
            )
        elif method == "pyemma_irrev":
            if 'N' in kwargs.keys(): kwargs.pop('N')
            _states, _stationary, _tmatrix = self.compute_pyemma_stationary(
                self.spliced_trajs, reversible=False, lag=self.lag  # TODO: Manage lag correctly
            )
        else:
            raise NotImplementedError("Invalid method specified")

        stationary = np.zeros(shape=self.n_stratified_clusters)
        tmatrix = np.zeros(
            shape=(self.n_stratified_clusters, self.n_stratified_clusters)
        )

        stationary[_states] = _stationary
        assert np.isclose(
            np.sum(stationary), 1.0
        ), "Stationary distribution not normalized!"
        tmatrix[np.ix_(_states, _states)] = _tmatrix

        mfpt = self.mfpt(
            self.mfpt_method,
            tmatrix,
            source_states,
            target_states,
            lag=self.lag,
            dt=self.dt,
            stationary=stationary,
        )

        if not np.isnan(mfpt):
            self.run.log_metric(
                f"set{set_idx}_{method}_mfpt_{self.current_direction}", mfpt
            )

        self.ness_df.loc[(set_idx, self.current_direction, method), :] = stationary

        return mfpt, tmatrix, stationary

    @staticmethod
    def mfpt(
            method,
            tmatrix,
            source_states,
            sink_states,
            dt,
            lag,
            stationary=None,
            clean_stationary=True,
    ):

        valid_methods = ["hill", "pyemma", "first_step", "fpt_distribution"]
        assert (
                method in valid_methods
        ), f"Invalid method -- choose one of {valid_methods}"

        _stationary = stationary.copy()
        _tmatrix = tmatrix.copy()

        # * Get MFPT via Hill relation
        # We shouldn't need to do this as a distribution, unless we don't actually have the stationary distribution
        #   -- the target flux should be constant in steady state.
        if method == "hill":
            # mfpt = cg.get_hill_mfpt(stationary, tmatrix, sink_states)
            if clean_stationary:
                _stationary[sink_states] = 0
                _stationary = _stationary / sum(_stationary)

            flux = 0
            for state in sink_states:
                flux += np.dot(_stationary, _tmatrix[:, state]).sum()

            mfpt = 1.0 / flux

        # * Get MFPT via PyEmma
        # To do this, we need a "proper" transition matrix -- i.e., not including any fully zeroed out rows
        elif method == "pyemma" or method == "first_step":
            # state_map = {}
            # _j = 0
            # for _i, _sum in enumerate(_tmatrix.sum(axis=1)):
            #     if np.isclose(_sum, 1):
            #         state_map[int(_i)] = int(_j)
            #         _j += 1
            #
            # try:
            #     remapped_source = np.array([
            #         state_map[x] for x in source_states if x in state_map.keys()
            #     ]).astype(int)
            #     remapped_target = np.array([
            #         state_map[x] for x in sink_states if x in state_map.keys()
            #     ]).astype(int)
            # except KeyError as e:
            #     raise e
            #
            # valid_states = list(state_map.keys())
            # clean_tmatrix = _tmatrix[valid_states][:, valid_states]

            # These methods fail by default for transition matrices with zero rows, because that'll fail deeptime's
            #   is_transition_matrix check.
            # However, by specifying a stationary distribution, it won't try to recalculate the stationary distribution,
            #   which is where that check is.
            # So even an "invalid" (by their definition) transition matrix will work.

            receiving_distribution = np.zeros_like(_stationary)
            receiving_distribution[source_states] = get_receiving_distribution(_tmatrix,
                                                                               _stationary, source_states)

            mfpt = deeptime.markov.tools.analysis.mfpt(_tmatrix, origin=source_states, target=sink_states,
                                                       mu=receiving_distribution)

        elif method == "fpt_distribution":

            assert _stationary is not None

            receiving_distribution = get_receiving_distribution(_tmatrix, _stationary, source_states)
            initial_probs = receiving_distribution
            initial_states = source_states
            # initial_probs = _stationary[source_states] / sum(_stationary[source_states])

            fpt_probs, _, _, times = MarkovFPT.adaptive_fpt_distribution(
                Tmatrix=_tmatrix,
                initial_states=initial_states,
                initial_state_probs=initial_probs,
                target_states=sink_states,
                verbose=False,
                fine_increment=1.1,
                increment=1.2,
            )

            mfpt = np.average(times, weights=fpt_probs)

        # Do NOT need to adjust for lag time here!
        mfpt = mfpt * dt  # \* lag

        return mfpt
