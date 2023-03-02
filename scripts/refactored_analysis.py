import sigopt
import os
import numpy as np
import logging
import deeptime
import tqdm.auto as tqdm
import msm_coarse_graining.traj_analysis as ta
from analysis import get_kl, AnalysisRun

log = logging.getLogger()

# Original set is 800x 200-step
metaparameters = dict(
    # trajectory_length=250,
    # n_trajectories_per_set=100,
    # n_trajectory_sets=4,
    trajectory_length=500,
    n_trajectories_per_set=200,
    n_trajectory_sets=4,
    stride=1,
    tic_rmsd=0,
    # For NESS
    unfolded_threshold=5,
    folded_threshold=-9,
    # For splicing recycling-distribution estimation
    splicing_msm_lag=None,
    # These below probably won't change much
    min_weight=1e-12,  # This just helps with numerical stability during the reweighting iteration
    last_frac=1.0,
    n_voelz_iters=1000,
    gen="2j0f-7-1ns",
    dataset_flags="verycoarse",
    mfpt_method="first_step",
    committor_direction='unfolding'
)

# MFPT reference
# TODO: Probably should just store this in the saved dataset, though that means I can't mess with thresholds
# log.info("Computing MFPT reference")
# stored_ref_data = np.load('/Users/russojd/Research/protein-like_msm/'
#                       'strict_strat_tic_cluster_lag-1ns_transition_matrix_ernesto_symm.npz', allow_pickle=True)
# stored_msm = stored_ref_data['msm'].item()
# stored_clusters = stored_ref_data['cluster_centers']
# fine_folded_states = np.argwhere(stored_clusters[:, 0] < metaparameters['folded_threshold']).flatten()
# fine_unfolded_states = np.argwhere(stored_clusters[:, 0] > metaparameters['unfolded_threshold']).flatten()
# metaparameters['ref_mfpt_folding'] = stored_msm.mfpt(fine_folded_states, fine_unfolded_states)
# metaparameters['ref_mfpt_unfolding'] = stored_msm.mfpt(fine_unfolded_states, fine_folded_states)

equil_methods = [
    "histogram",
    "naive",
    "pyemma_irrev",
    "pyemma_rev",
    "reweighted",
]
mfpt_methods = [
    "naive",
    "pyemma_irrev",
    "pyemma_rev",
    "reweighted",
]

if __name__ == "__main__":

    experiment = sigopt.create_experiment(
        name=f"2JOF - {metaparameters['n_trajectory_sets']} set, Lower data",
        budget=50,
        parameters=[
            dict(name='N', type='int', bounds={'min': 2, 'max': metaparameters['trajectory_length']-1 }),
            dict(name='last_frac', type='double', bounds={'min': 0.25, 'max': 1.0 }),
            dict(name='lag', type='int', bounds={'min': 1, 'max': metaparameters['trajectory_length']-5 }),
            # dict(
            #     name="N", type="int", grid=[2, 5, 10, 25, 50, 100, 150, 200, 300, 400]
            # ),
            # dict(
            #     name="lag",
            #     type="int",
            #     grid=[-1, 1],
            #     grid=[1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 400],
            # ),
        ],
        metrics=[
            # {"name": "std_set_voelz_kl", "strategy": "store"},
            {
                "name": "lag_vs_N",
                "strategy": "constraint",
                "objective": "minimize",
                "threshold": 1.0,
            },
            {
                "name": "avg_set_voelz_kl",
                # "strategy": "store",
                "strategy": "optimize",
                "objective": "minimize",
            },
            {
                "name": "set_committor_kl_std_reweighted",
                # "strategy": "store",
                "strategy": "optimize",
                "objective": "minimize",
            },
            {"name": "set_committor_kl_mean_reweighted", "strategy": "store"},
            {"name": "set_committor_kl_mean_pyemma_irrev", "strategy": "store"},
            {"name": "set_committor_kl_std_pyemma_irrev", "strategy": "store"},
            {
                "name": "set_voelz_unfolding_mfpt_dev",
                "strategy": "store",
                # "strategy": "optimize",
                # "objective": "minimize",
            },
            {
                "name": "set_voelz_folding_mfpt_dev",
                # "strategy": "optimize",
                "strategy": "store",
                "objective": "minimize",
            },
            {"name": "avg_folding_reweighted_mfpt", "strategy": "store"},
            {"name": "avg_unfolding_reweighted_mfpt", "strategy": "store"},
            {"name": "avg_folding_resliced_mfpt", "strategy": "store"},
            {"name": "avg_unfolding_resliced_mfpt", "strategy": "store"},
            {"name": "avg_folding_pyemma_irrev_mfpt", "strategy": "store"},
            {"name": "avg_unfolding_pyemma_irrev_mfpt", "strategy": "store"},
        ],
        metadata=metaparameters,
    )

    # Prepare the dataset, if necessary
    gen_path = f'../../generated/{metaparameters["gen"]}'
    traj_set_name = (
        f"{metaparameters['gen']}_"
        f"{metaparameters['n_trajectory_sets']}-"
        f"{metaparameters['n_trajectories_per_set']}-"
        f"{metaparameters['trajectory_length']}"
    )

    # !  0. If subsampling, check if the desired subsampled data already exists, and if so load it
    if os.path.exists(f"{gen_path}/{traj_set_name}.npz"):
        log.info(f"Trajectories already exist, using {traj_set_name}.npz")
        trajectory_data = np.load(f"{gen_path}/{traj_set_name}.npz")

    # ! Otherwise, we'll need to create it
    else:
        # !   1. Load up the full dataset
        stored_tica_data = np.load(f"{gen_path}/tica.npz")
        stored_traj_data = np.load(
            f'{gen_path}/rediscretized-{metaparameters["dataset_flags"]}.npz'
        )

        # !   2. Select trajectories
        projection = stored_tica_data["projection"]
        # *   2a. Subsample as desired, to reduce the number of trajectories
        # First, get the original number of trajectories
        # Then, using the desired number of sets, get the number of trajectories per set
        # Then get the desired number of trajectories per set

        # I.e., for 800 original trajectories in 4 sets, this gives 4 trajectories per set
        original_n_trajectories_per_set = projection.shape[0] // metaparameters['n_trajectory_sets']
        # Continuing with the above example, if you wanted 200 trajectories per set, this would give subsample_trajs=1
        subsample_trajs = int(
            original_n_trajectories_per_set / metaparameters["n_trajectories_per_set"]
        )
        # *   2b. Truncate to reduce the length of trajectories
        stratified_projection_sets = [
            projection[
                i :: metaparameters["n_trajectory_sets"] * subsample_trajs,
                : metaparameters["trajectory_length"]
                * metaparameters["stride"] : metaparameters["stride"],
            ]
            for i in range(metaparameters["n_trajectory_sets"])
        ]

        # !   3. Clean
        # TODO: Refactor this out to its own function, or file
        bin_boundaries = stored_traj_data["bin_boundaries"]
        stratified_centers = stored_traj_data["combined_kmeans_centers"]

        n_trap_states = np.inf
        good_states = np.arange(stratified_centers.shape[0])
        traps = []
        traps_to_remove = []
        clean_centers = stratified_centers.copy()[good_states]

        rng = np.random.RandomState(seed=1337)

        for projection_set in stratified_projection_sets:

            while n_trap_states > 0:

                print("********")

                # Clean (does nothing first iter)
                # Note that good_states should be indexed over the same
                good_states = np.setdiff1d(good_states, traps_to_remove)
                clean_centers = clean_centers[good_states]

                # Discretize
                discretized_trajectories, _, _ = ta.transform_stratified(
                    clean_centers,
                    [[np.arange(len(p)), p.T] for p in projection_set],
                    bin_boundaries,
                    metaparameters["tic_rmsd"],
                )
                discretized_trajectories = np.array(discretized_trajectories).astype(
                    int
                )

                # Build count model
                count_model = deeptime.markov.TransitionCountEstimator(
                    count_mode="sliding", lagtime=1
                ).fit_fetch(discretized_trajectories)
                good_states = np.arange(count_model.count_matrix.shape[0])

                # Identify trap states
                traps, (empty, sink, source, disjoint) = ta.find_traps(
                    count_model.count_matrix
                )

                # Only track non-empty traps -- we'll always remove all empty states
                # - These track indices of trap states in the count matrix
                traps = np.setdiff1d(traps, empty)
                print(traps, len(traps), empty)

                if len(traps) == 0:
                    print("No traps remain")
                    break

                traps_to_remove = set()
                for i, (bin_lower, bin_upper) in tqdm.tqdm(
                    enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])),
                    total=len(bin_boundaries) - 1,
                ):

                    # Make a KMeans estimator using only the cluster centers in this bin
                    trap_idxs_in_bin = np.argwhere(
                        (clean_centers[traps, metaparameters["tic_rmsd"]] >= bin_lower)
                        & (clean_centers[traps, metaparameters["tic_rmsd"]] < bin_upper)
                    ).flatten()

                    if len(trap_idxs_in_bin) == 0:
                        continue

                    traps_in_bin = traps[trap_idxs_in_bin]

                    # Randomly choose a state for removal with probability inversely proportional to the number of counts
                    trap_probabilities = 1 / count_model.state_histogram[traps_in_bin]
                    trap_probabilities /= sum(trap_probabilities)
                    trap_to_remove = rng.choice(traps_in_bin, p=trap_probabilities)

                    print(f"Traps in bin {i}: {traps[trap_idxs_in_bin]}")
                    print(
                        f"\t Bin {i} has {len(trap_idxs_in_bin)} traps with populations {count_model.state_histogram[traps_in_bin]}"
                    )
                    print(f"\t Selected trap {trap_to_remove} for removal this round")

                    traps_to_remove.add(trap_to_remove)

                traps_to_remove.update([x for x in empty.flatten()])
                # Count number of remaining trap states
                n_trap_states = len(traps)
                traps_to_remove = np.array([x for x in traps_to_remove])

        stratified_centers = clean_centers
        stratified_center_sort = np.argsort(
            stratified_centers[:, metaparameters["tic_rmsd"]]
        )
        n_stratified_clusters = stratified_centers.shape[0]
        print(f"Have {n_stratified_clusters} after cleaning")

        # Now discretize all, according to the new cleaning
        stratified_trajectory_sets = []
        for projection_set in stratified_projection_sets:
            # Discretize
            discretized_trajectories, _, _ = ta.transform_stratified(
                clean_centers,
                [[np.arange(len(p)), p.T] for p in projection_set],
                bin_boundaries,
                metaparameters["tic_rmsd"],
            )
            discretized_trajectories = np.array(discretized_trajectories).astype(int)

            count_model = deeptime.markov.TransitionCountEstimator(
                count_mode="sliding", lagtime=1
            ).fit_fetch(discretized_trajectories)
            good_states = np.arange(count_model.count_matrix.shape[0])

            # Identify trap states
            traps, (empty, sink, source, disjoint) = ta.find_traps(
                count_model.count_matrix
            )
            assert len(traps) == 0

            stratified_trajectory_sets.append(discretized_trajectories)

        # Recalculate the reference distribution
        og_stratified_centers = stored_traj_data["combined_kmeans_centers"]
        # This maps OLD : NEW
        center_remapping, _, _ = ta.transform_stratified(
            clean_centers,
            [[np.arange(len(og_stratified_centers)), og_stratified_centers.T]],
            bin_boundaries,
            metaparameters["tic_rmsd"],
        )
        new_reference_distribution = np.zeros(n_stratified_clusters)
        for old_cluster, new_cluster in enumerate(center_remapping[0].astype(int)):
            new_reference_distribution[new_cluster] += stored_traj_data[
                "remapped_reference_distribution"
            ][old_cluster]
        remapped_reference = new_reference_distribution

        assert np.isclose(
            sum(new_reference_distribution), 1.0
        ), "New reference not normalized"

        # !   4. Save the cleaned dataset
        log.info(
            f"Finished cleaning. Saving this dataset as {gen_path}/{traj_set_name}.npz"
        )
        np.savez(
            f"{gen_path}/{traj_set_name}.npz",
            stratified_trajectory_sets=stratified_trajectory_sets,
            center_remapping=center_remapping,
            new_reference_distribution=new_reference_distribution,
            stratified_centers=stratified_centers,
            good_states=good_states,
        )

    loaded_dataset = np.load(f"{gen_path}/{traj_set_name}.npz")
    stratified_trajectory_sets = loaded_dataset["stratified_trajectory_sets"]
    stratified_centers = loaded_dataset["stratified_centers"]
    new_reference_distribution = loaded_dataset["new_reference_distribution"]
    n_stratified_clusters = stratified_centers.shape[0]

    # TODO: Verify all parameters (particularly lagtime, last_frac) are being used correctly / actually being used
    for run in experiment.loop():
        with run:
            try:
                run.log_dataset(
                    metaparameters["gen"] + "-" + metaparameters["dataset_flags"]
                )

                if metaparameters['last_frac'] is None:
                    metaparameters['last_frac'] = run.params['last_frac']

                # Lag time must be shorter than N, so this must be below 0
                lag_vs_N = run.params["lag"] / run.params["N"]
                run.log_metric("lag_vs_N", lag_vs_N)
                if lag_vs_N >= 1.0:
                    continue

                if run.params['lag'] == -1:
                    continue


                # TODO: What's the appropriate dt for the MFPTs?
                analysis_run = AnalysisRun(
                    run,
                    new_reference_distribution,
                    stratified_trajectory_sets,
                    dt=1e-3,  # so MFPTs are in units of µs
                    lag=run.params["lag"],
                    metaparameters=metaparameters,
                )

                # ! Estimate Equilibrium
                log.info("Computing equilibrium")
                # analysis_run.current_direction = 'equilibrium'

                for traj_set_idx, trajectories in enumerate(stratified_trajectory_sets):
                    analysis_run.current_traj_set = traj_set_idx
                    analysis_run.active_df = analysis_run.equil_df

                    for _method in equil_methods:
                        kl = analysis_run.compute_stationary(
                            _method, lag=run.params.get("lag"), N=run.params.get("N")
                        )

                analysis_run.equil_df.to_pickle(f"../results/{run.id}_equil_df.pkl")

                # If we only have one trajectory set, the set-set KL divergence is not meaningful
                if metaparameters["n_trajectory_sets"] > 1:
                    mean_kl, std_kl = analysis_run.compute_avg_kl("reweighted")
                    if np.isnan(mean_kl):
                        run.log_failure()
                    else:
                        run.log_metric(f"avg_set_voelz_kl", mean_kl)
                        run.log_metric(f"std_set_voelz_kl", std_kl)
                else:
                    run.log_metric(f"avg_set_voelz_kl", 0)
                    run.log_metric(f"std_set_voelz_kl", 0)

                # ! Estimate MFPT
                mfpt_estimates = {}
                for traj_set_idx, trajectories in enumerate(stratified_trajectory_sets):
                    mfpt_estimates[traj_set_idx] = {}

                    analysis_run.current_traj_set = traj_set_idx
                    analysis_run.active_df = analysis_run.ness_df

                    # * Trajectory splicing from equilibrium to NESS
                    # Pick out folded and unfolded states
                    folded_states = np.argwhere(
                        stratified_centers[:, 0] < metaparameters["folded_threshold"]
                    ).flatten()
                    unfolded_states = np.argwhere(
                        stratified_centers[:, 0] > metaparameters["unfolded_threshold"]
                    ).flatten()

                    for (direction, source_states, sink_states) in [
                        ("unfolding", folded_states, unfolded_states),
                        ("folding", unfolded_states, folded_states),
                    ]:
                        mfpt_estimates[traj_set_idx][direction] = {}
                        analysis_run.current_direction = direction

                        # Create the spliced states
                        # analysis_run.splice_trajectories(
                        #     sink_states=sink_states, source_states=source_states
                        # )
                        analysis_run.iterative_trajectory_splicing(
                            sink_states=sink_states, source_states=source_states,
                            # msm_lag=metaparameters['splicing_msm_lag'], msm_reversible=False,
                            splice_msm_lag=run.params.get("lag"), msm_reversible=False,
                            convergence=1e-9, max_iterations=100
                        )

                        for method in mfpt_methods:
                            _mfpt, _, _stationary = analysis_run.compute_mfpt(
                                method,
                                source_states=source_states,
                                target_states=sink_states,
                                # lag=run.params.get("lag"),
                                N=run.params.get("N"),
                            )

                            if method == "reweighted":
                                if _mfpt is not None:
                                    mfpt_estimates[traj_set_idx][direction][
                                        "resliced"
                                    ] = _mfpt[0]
                                    mfpt_estimates[traj_set_idx][direction][
                                        "reweighted"
                                    ] = _mfpt[1]
                                else:
                                    mfpt_estimates[traj_set_idx][direction]["resliced"] = np.nan
                                    mfpt_estimates[traj_set_idx][direction]["reweighted"] = np.nan

                            else:
                                mfpt_estimates[traj_set_idx][direction][method] = _mfpt

                for direction in ["unfolding", "folding"]:
                    for method in ["pyemma_irrev", "resliced", "reweighted"]:

                        _mfpt_estimates = [
                            traj_set_estimates[direction][method]
                            for traj_set_idx, traj_set_estimates in mfpt_estimates.items()
                        ]

                        set_avg_mfpt_estimate = np.nanmean(_mfpt_estimates)
                        set_std_mfpt_estimate = np.nanstd(_mfpt_estimates) / np.sqrt(len(_mfpt_estimates))
                        run.log_metric(f"avg_{direction}_{method}_mfpt", set_avg_mfpt_estimate)
                        run.log_metric(f"set_voelz_{direction}_mfpt_dev", set_std_mfpt_estimate)

                analysis_run.ness_df.to_pickle(f"../results/{run.id}_ness_df.pkl")

                # ! Calculate Committor
                # Do this using the NESS and Equil distributions in equil_df and ness_df

                # Dataframes are indexed by:
                #   - equil_df.loc[(set idx, method), state]
                #   - ness_df.loc[(set idx, direction, method), state]

                # committor_estimates = np.full(shape=(len(stratified_trajectory_sets), n_stratified_clusters),
                #                               fill_value=np.nan)
                # for traj_set_idx, trajectories in enumerate(stratified_trajectory_sets):
                #     for direction in ["unfolding", "folding"]:
                #         for method in ["pyemma_irrev", "resliced", "reweighted"]:
                #
                #             _equil = analysis_run.equil_df.loc[(traj_set_idx, method)]
                #             _ness = analysis_run.ness_df.loc[(traj_set_idx, direction, method)]
                #
                #             _committors = _equil / _ness
                #             committor_estimates[traj_set_idx] = _committors

                # * Now compute similarity across committors and log it as a metric
                # AKA, get a set-set KL for the pyemma and the reweighted committors

                for method in ['pyemma_irrev', 'reweighted']:
                    # for direction in ["unfolding", "folding"]:
                    direction = metaparameters['committor_direction']

                    if direction == 'unfolding':
                        source_states = folded_states
                        sink_states = unfolded_states
                    elif direction == 'folding':
                        source_states = unfolded_states
                        sink_states = folded_states
                    else:
                        raise NotImplementedError

                    transition_bins = np.setdiff1d(
                        np.arange(n_stratified_clusters),
                        np.concatenate([source_states, sink_states]),
                    )


                    # TODO: fold this back into analysis_run.get_avg_kl
                    committor_estimates = np.full(shape=(len(stratified_trajectory_sets), n_stratified_clusters),
                                                  fill_value=np.nan)
                    for traj_set_idx, trajectories in enumerate(stratified_trajectory_sets):
                        _equil = analysis_run.equil_df.loc[(traj_set_idx, method)]
                        _ness = analysis_run.ness_df.loc[(traj_set_idx, direction, method)]

                        # NESS and Equilibrium are both normalized -- however, the committor depends on the
                        #   two directional NESS adding up to equilibrium (which can't happen if they're both
                        #   independently normalized)
                        #   This renormalizes the NESS distribution to match the equilibrium distribution in the
                        #   source state. Nominally, after this renormalization, NESS should match equilibrium
                        #   in the source.
                        normalization_constant = _ness[source_states].sum() / _equil[source_states].sum()

                        # _committors = (_ness / normalization_constant) / _equil
                        # Divide like this, to avoid getting NaN in case equilibrium has a bin with 0
                        _committors = np.divide(
                            _ness / normalization_constant,
                            _equil,
                            out=np.zeros_like(_ness),
                            where=_equil > 0
                        )

                        # Cap committors at a value of 1
                        _committors[_committors > 1] = 1

                        committor_estimates[traj_set_idx] = _committors

                    kls = np.full(
                        shape=(
                            analysis_run.metaparameters["n_trajectory_sets"],
                            analysis_run.metaparameters["n_trajectory_sets"],
                        ),
                        fill_value=np.nan,
                    )



                    for x, y in np.array(
                            np.triu_indices(analysis_run.metaparameters["n_trajectory_sets"], 1)
                    ).T:

                        setA_committors = committor_estimates[x].astype(float)
                        setB_committors = committor_estimates[y].astype(float)
                        kl_sum = get_kl(setA_committors[transition_bins], setB_committors[transition_bins], return_nan=True)
                        kls[x, y] = kl_sum

                    mean_kl = np.nanmean(kls)
                    std_kl = 2 * np.nanstd(kls) / np.sqrt(analysis_run.metaparameters["n_trajectory_sets"])
                    run.log_metric(f"set_committor_kl_std_{method}", std_kl)
                    run.log_metric(f"set_committor_kl_mean_{method}", mean_kl)

            except (ValueError, OverflowError, TypeError) as e:
                log.error(e)
                run.log_failure()
                raise e
            except np.linalg.LinAlgError as e:
                log.error(e)
                run.log_failure()


    if True:

        # ! Global model
        # Now, pick the best parameters and build a global model.
        # "Best" goes by our heuristic of "lowest average set-set KL divergence"
        best_run = list(experiment.get_best_runs())[0]
        best_N = best_run.assignments["N"]
        best_lag = best_run.assignments["lag"]
        best_last_N = best_run.assignments["last_N"]

        # best_N = 75
        # best_lag = 1

        full_metaparameters = {k: v for k, v in metaparameters.items()}
        full_metaparameters["n_trajectory_sets"] = 1
        full_metaparameters["n_trajectories_per_set"] = 800

        del run, metaparameters
        with sigopt.create_run() as optimal_run:

            optimal_run.params["N"] = best_N
            optimal_run.params["lag"] = best_lag
            optimal_run.params['last_N'] = best_last_N

            optimal_run.metadata = full_metaparameters
            optimal_run.log_dataset(
                full_metaparameters["gen"] + "-" + full_metaparameters["dataset_flags"]
            )

            #! Load up global data
            traj_set_name = (
                f"{full_metaparameters['gen']}_"
                f"{full_metaparameters['n_trajectory_sets']}-"
                f"{full_metaparameters['n_trajectories_per_set']}-"
                f"{full_metaparameters['trajectory_length']}"
            )

            assert os.path.exists(f"{gen_path}/{traj_set_name}.npz"), "Global model trajectories don't exist"

            log.info(f"Using {traj_set_name}.npz")
            gen_path = f'../generated/{full_metaparameters["gen"]}'
            loaded_dataset = np.load(f"{gen_path}/{traj_set_name}.npz")
            stratified_trajectory_sets = loaded_dataset["stratified_trajectory_sets"]
            stratified_centers = loaded_dataset["stratified_centers"]
            new_reference_distribution = loaded_dataset["new_reference_distribution"]
            n_stratified_clusters = stratified_centers.shape[0]

            analysis_run = AnalysisRun(
                optimal_run,
                new_reference_distribution,
                stratified_trajectory_sets,
                dt=1e-3,  # so MFPTs are in units of µs
                lag=optimal_run.params["lag"],
                metaparameters=full_metaparameters,
            )

            #! Estimate equilibrium
            analysis_run.current_traj_set = 0
            analysis_run.active_df = analysis_run.equil_df

            for _method in equil_methods:
                # This will also log KL-divergence to reference

                kl = analysis_run.compute_stationary(
                    _method,
                    lag=optimal_run.params.get("lag"),
                    N=optimal_run.params.get("N"),
                )

            analysis_run.equil_df.to_pickle(f"../results/{optimal_run.id}_equil_df.pkl")

            # ! Estimate MFPT
            analysis_run.current_traj_set = 0
            analysis_run.active_df = analysis_run.ness_df

            # * Trajectory splicing from equilibrium to NESS
            # Pick out folded and unfolded states
            folded_states = np.argwhere(
                stratified_centers[:, 0] < full_metaparameters["folded_threshold"]
            ).flatten()
            unfolded_states = np.argwhere(
                stratified_centers[:, 0] > full_metaparameters["unfolded_threshold"]
            ).flatten()

            for (direction, source_states, sink_states) in [
                ("unfolding", folded_states, unfolded_states),
                ("folding", unfolded_states, folded_states),
            ]:
                analysis_run.current_direction = direction

                # Create the spliced states
                analysis_run.splice_trajectories(
                    sink_states=sink_states, source_states=source_states
                )

                for method in mfpt_methods:
                    # This also logs the MFPT
                    _mfpt, _, _ = analysis_run.compute_mfpt(
                        method,
                        source_states=source_states,
                        target_states=sink_states,
                        lag=optimal_run.params.get("lag"),
                        N=optimal_run.params.get("N"),
                        last_frac=optimal_run.params.get("last_frac")
                    )

            analysis_run.ness_df.to_pickle(f"../results/{optimal_run.id}_ness_df.pkl")

            log.info(f"Completed optimized global run {optimal_run.id}")
