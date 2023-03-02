#!/usr/bin/env python

"""
Tests for `msm_coarse_graining` analysis.
This verifies that the analyses are working correctly.
"""

import pytest
import numpy as np
from mr_toolkit.coarse_graining import msm_coarse_graining as msm_cg
import msmtools


def test_microscopic_eqm(ref_tridiag, ref_tridiag_equil):
    """
    This test verifies the microscopic equilibrium distribution is correctly recovered by explicitly testing
    that the transition matrix is stationary under it.
    """

    equil = msm_cg.get_equil(ref_tridiag)

    assert np.isclose(equil, ref_tridiag_equil).all()


@pytest.mark.xfail
def test_hill_mfpt(ref_tridiag):
    """
    This test verifies the Hill-relation based MFPT calculation
    """
    target_state = len(ref_tridiag) - 1
    source_state = 0

    t_ss = msm_cg.make_ss(ref_tridiag, [target_state], [source_state], keep_sink_entry=True)
    ss_dist = msm_cg.get_equil(t_ss)

    assert np.all(np.isclose(ss_dist, msmtools.analysis.stationary_distribution(t_ss)))

    mfpt = msm_cg.get_hill_mfpt(ss_dist, t_ss, [target_state])

    naive_mfpt = msm_cg.get_naive_hill_mfpt(t_ss, ss_dist, [5], [0,1,2,3,4])

    msmtools_mfpt = msmtools.analysis.mfpt(t_ss, target=[target_state], origin=[source_state])

    assert np.all(np.isclose(mfpt, msmtools_mfpt)), (mfpt, msmtools_mfpt)


@pytest.mark.skip
def test_microscopic_committors():
    """
    This test verifies the microscopic committors are correctly recovered by explicitly testing that the
    average first-step committor gives back the initial bin committor.
    """

    assert False
