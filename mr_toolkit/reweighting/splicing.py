"""
Code for splicing equilibrium trajectories into nonequilibrium steady-state trajectories.
"""
import pyemma
import tqdm.auto as tqdm
import numpy as np

import logging

log = logging.getLogger()


def splice_trajectory(
        trajectory,
        splice_trajectories,
        target_states,
        recycling_states,
        recycling_probabilities,
        rng,
        target_steps_to_keep=1,
):
    points_in_target = np.isin(trajectory, target_states)
    splice_point = None

    if not points_in_target.any():
        return trajectory, None

    first_target_entry = np.argmax(points_in_target)

    # TODO: What do we want to do with trajectories that start in this state? Splice as normal?
    if first_target_entry == 0:
        pass

    # This preserves one point in the target
    splice_point = first_target_entry + target_steps_to_keep

    # If the target entries occur within steps_to_keep of the end of the trajectory, don't splice anything
    if splice_point >= len(trajectory) - target_steps_to_keep + 1:
        return trajectory, None

    steps_remaining = len(trajectory) - first_target_entry - target_steps_to_keep

    # Loop here, because you might pick a splice state that's not present within :steps_remaining of any traj
    # TODO: Maybe replace this by looping over the whole splice procedure, so my selection is "pure" every time.
    points_in_state = []
    while len(points_in_state) == 0:
        # Choose a state to splice from
        splice_state = rng.choice(recycling_states, p=recycling_probabilities)

        # Now choose a point in that state.
        # Only choose where you'll have enough trajectory to fully splice what you need
        points_in_state = np.argwhere(
            splice_trajectories[:, :-steps_remaining] == splice_state
        )

    traj_idx, point_idx = rng.choice(points_in_state, axis=0)

    log.debug(f"Splicing a trajectory at point {splice_point}, "
              f"replacing {trajectory[splice_point]} with state {splice_state}")

    spliced_trajectory = np.concatenate(
        [
            trajectory[:splice_point],
            splice_trajectories[traj_idx, point_idx: point_idx + steps_remaining],
        ]
    )

    return spliced_trajectory, splice_point


def get_receiving_distribution(tmatrix, stationary, source_states):
    # All transition matrix elements into the folded states
    source_boundary_states = np.argwhere(
        tmatrix[:, source_states].sum(axis=1)
    ).flatten()

    # Now filter down to the states that aren't already IN the source state
    source_boundary_states_exclusive = np.setdiff1d(
        source_boundary_states, source_states
    )

    boundary_probabilities = stationary[
        source_boundary_states_exclusive
    ]

    flux_into_source = boundary_probabilities @ \
                       tmatrix[source_boundary_states_exclusive][:, source_states]

    source_receiving_distribution = flux_into_source / sum(flux_into_source)

    return source_receiving_distribution


def splice_trajectories(
        trajs_to_splice,
        source_states,
        sink_states,
        n_clusters,
        msm_lag=1,
        msm_reversible=False,
        target_steps_to_keep=1,
        pbar_visible=True
):
    # Build an MSM to approximate the equilibrium distribution over the boundary states
    # TODO: Do we want to just use the PyEmma MSM? Or is there a better choice?

    pyemma_msm = pyemma.msm.estimate_markov_model(
        [x for x in trajs_to_splice],
        lag=msm_lag,
        reversible=msm_reversible,
    )

    tmatrix = np.zeros(shape=(n_clusters, n_clusters))
    tmatrix[np.ix_(pyemma_msm.active_set, pyemma_msm.active_set)] = pyemma_msm.transition_matrix
    stationary = np.zeros(n_clusters)
    stationary[pyemma_msm.active_set] = pyemma_msm.stationary_distribution

    recycling_states = source_states
    recycling_probabilities = get_receiving_distribution(tmatrix, stationary, source_states)

    spliced_trajs = np.array(
        [[t for t in traj] for traj in trajs_to_splice]
    )

    # TODO: Keep looping over the trajectories until no more splicing happens
    #   This is necessary because it's possible to splice a fragment that has a sink entry itself, in which case
    #   you'd need to splice again.
    #   Right now, this is handled by only splicing fragments that don't re-enter the target in `splice_trajectory`
    #   The problem with doing that is that on the second pass, all the spliced trajectories now look like they
    #       entered the target at step 1, and so they try to get spliced again. This can be fixed by storing the indices
    #       of any spliced trajectories, and their splice point, and only splicing again if the new splice point is
    #       later.
    rng = np.random.default_rng(seed=42)
    did_splicing = 0
    for i, trajectory in tqdm.tqdm(
            enumerate(spliced_trajs), total=len(spliced_trajs), desc="Splicing",
            disable=not pbar_visible
    ):

        spliced_trajectory, splice_point = splice_trajectory(
            trajectory,
            rng=rng,
            # Splice using the original set of trajectories, rather than our updating spliced ones
            splice_trajectories=trajs_to_splice,
            target_states=sink_states,
            recycling_states=recycling_states,
            recycling_probabilities=recycling_probabilities,
            target_steps_to_keep=target_steps_to_keep,
        )

        if splice_point is not None:
            did_splicing += 1

        spliced_trajs[i] = spliced_trajectory

    # spliced_trajs = spliced_trajs
    return spliced_trajs


def iterative_trajectory_splicing(
        trajs,
        source_states,
        sink_states,
        n_clusters,
        splice_msm_lag=1,
        msm_reversible=False,
        target_steps_to_keep=1,
        convergence=1e-9,
        max_iterations=100):

    spliced_trajs = splice_trajectories(
        trajs_to_splice=trajs,
        msm_lag=splice_msm_lag,
        msm_reversible=msm_reversible,
        n_clusters=n_clusters,
        target_steps_to_keep=target_steps_to_keep,
        sink_states=sink_states.flatten(),
        source_states=source_states.flatten(),
        pbar_visible=False
        )

    pyemma_ness_msm = pyemma.msm.estimate_markov_model(
        [x for x in spliced_trajs],
        lag=splice_msm_lag,
        reversible=msm_reversible,
    )
    original_ness = np.zeros(n_clusters)
    original_ness[pyemma_ness_msm.active_set] = pyemma_ness_msm.stationary_distribution.copy()
    previous_ness = original_ness.copy()

    for _iteration in tqdm.trange(max_iterations, desc="Splicing iteration"):
        spliced_trajs = splice_trajectories(
                            trajs_to_splice=spliced_trajs.copy(),
                            msm_lag=splice_msm_lag, msm_reversible=msm_reversible,
                            n_clusters=n_clusters,
                            target_steps_to_keep=target_steps_to_keep,
                            sink_states=sink_states.flatten(),
                            source_states=source_states.flatten(),
                            pbar_visible=False)

        pyemma_ness_msm = pyemma.msm.estimate_markov_model(
            [x for x in spliced_trajs],
            lag=splice_msm_lag,
            reversible=msm_reversible,
        )

        new_ness = np.zeros(n_clusters)
        new_ness[pyemma_ness_msm.active_set] = pyemma_ness_msm.stationary_distribution

        rms_change_from_original = np.sqrt(np.mean(np.power(new_ness - original_ness, 2)))
        rms_change_from_last = np.sqrt(np.mean(np.power(new_ness - previous_ness, 2)))
        log.debug(
            f"RMS change at iter {_iteration} is {rms_change_from_original:.2e} from original, "
            f"{rms_change_from_last:.2e} from previous")

        previous_ness = new_ness.copy()

        if rms_change_from_last < convergence:
            log.info(f"Splicing converged after {_iteration} iterations")
            break

    return spliced_trajs
