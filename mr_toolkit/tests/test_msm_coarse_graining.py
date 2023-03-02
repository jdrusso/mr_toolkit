#!/usr/bin/env python

"""
Tests for `msm_coarse_graining` package.
These are just tests on the basic coarse-graining procedure itself.
"""

from mr_toolkit.coarse_graining import msm_coarse_graining as msm_cg
import numpy as np


def test_build_tridiagonal(ref_tridiag):
    """
    Checks that a tridiagonal matrix was constructed with the appropriate ratio of self to neighbor transition probabilities.
    """

    tridiag = msm_cg.build_fine_transition_matrix(3, 6)

    assert np.all(tridiag == ref_tridiag)


def test_bin_weight_averaging_onestep(ref_avg_bin_weight, ref_tridiag, initial_weights):
    """
    Computes the time-avged weights \bar{w}_i for a trajectory length of 2 steps (S=1).
    """

    bin_weights = msm_cg.compute_avg_bin_weights(
        initial_weights, transition_matrix=ref_tridiag, max_s=1
    )

    assert np.all(np.isclose(bin_weights, initial_weights))


def test_bin_weight_averaging_twostep(
    ref_avg_bin_weight, ref_tridiag, initial_weights, ref_twostep_weights
):
    """
    Computes the time-avged weights \bar{w}_i for a trajectory length of 3 steps (S=2).
    """

    bin_weights = msm_cg.compute_avg_bin_weights(
        initial_weights, transition_matrix=ref_tridiag, max_s=2
    )

    assert np.all(np.isclose(bin_weights, ref_twostep_weights))


def test_bin_weight_averaging_tenstep(
    ref_avg_bin_weight, ref_tridiag, initial_weights, ref_tenstep_weights
):
    """
    Computes the time-avged weights \bar{w}_i for a trajectory length of 11 steps (S=10).
    """

    bin_weights = msm_cg.compute_avg_bin_weights(
        initial_weights, transition_matrix=ref_tridiag, max_s=10
    )

    assert np.all(np.isclose(bin_weights, ref_tenstep_weights))


def test_bin_weight_averaging_tenstep_lag5(
    ref_avg_bin_weight, ref_tridiag, initial_weights, ref_tenstep_lag5_weights
):
    """
    Computes the time-avged weights \bar{w}_i for a trajectory length of 11 steps (S=10) at lag=5.
    """

    bin_weights = msm_cg.compute_avg_bin_weights(
        initial_weights, transition_matrix=ref_tridiag, max_s=10, lag=5
    )

    assert np.all(np.isclose(bin_weights, ref_tenstep_lag5_weights))


def test_coarse_graining_uniform_weights(
    ref_tridiag, ref_coarse_grained_uniform_weights, ref_cg_map
):
    """
    Checks that a simple tridiagonal matrix is correctly coarse-grained, according to uniform weights.
    """
    weights = np.array([1, 1, 1, 1, 1, 1]) / 6

    coarse_grained = msm_cg.coarse_grain(ref_tridiag, ref_cg_map, weights)

    assert np.all(coarse_grained == ref_coarse_grained_uniform_weights), (coarse_grained, ref_coarse_grained_uniform_weights)


def test_coarse_graining_nonuniform_weights(
    ref_tridiag, ref_coarse_grained_nonuniform_weights, initial_weights, ref_cg_map
):
    """
    Checks that a simple tridiagonal matrix is correctly coarse-grained under nonuniform weights.
    """

    coarse_grained = msm_cg.coarse_grain(ref_tridiag, ref_cg_map, initial_weights)

    assert np.all(np.isclose(coarse_grained, ref_coarse_grained_nonuniform_weights))

def test_coarse_graining_timeavg_weights(
    ref_tridiag,
    ref_coarse_grained_timeavged_weights,
    initial_weights,
    ref_tenstep_lag5_weights,
    ref_cg_map,
):
    """
    Check that a simple tridiagonal matrix, and some nonuniform initial weights, produce the appropriate coarse-grained
    matrix.

    Todo
    ----
    Need to pass a P_lambda computed at a lag time that isn't 1. This is consistent w/ my manual calculation, if the
    original tridiagonal is also the matrix at a lag of 5.
    """

    avg_bin_weights = msm_cg.compute_avg_bin_weights(
        initial_weights, transition_matrix=ref_tridiag, max_s=10, lag=5
    )

    assert np.all(np.isclose(avg_bin_weights, ref_tenstep_lag5_weights))

    coarse_grained = msm_cg.coarse_grain(
        ref_tridiag, ref_cg_map, avg_bin_weights, lag=5
    )

    assert np.all(np.isclose(coarse_grained, ref_coarse_grained_timeavged_weights))


def test_building_occupancy_th2(ref_occ_th2, ref_tridiag, initial_weights, ref_cg_map):
    """
    Tests constructing an occupancy matrix at with S=10 and a time horizon of 2.

    Note that a time horizon of 1 is just coarse-graining the MSM at lag 1, with no averaging.
    """

    occ_result = msm_cg.build_occupancy(
        ref_tridiag, initial_weights, ref_cg_map, s=10, time_horizon=2
    )

    assert np.all(np.isclose(occ_result, ref_occ_th2))

def test_building_occupancy_th5(ref_occ_th5, ref_tridiag, initial_weights, ref_cg_map):
    """
    Tests constructing an occupancy matrix at with S=10 and a time horizon of 5.
    """

    occ_result = msm_cg.build_occupancy(
        ref_tridiag, initial_weights, ref_cg_map, s=10, time_horizon=5
    )

    assert np.all(np.isclose(occ_result, ref_occ_th5))
