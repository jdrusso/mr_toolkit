API Documentation
=================

.. autosummary::

   mr_toolkit.coarse_graining.msm_coarse_graining
   mr_toolkit.reweighting.analysis
   mr_toolkit.reweighting.splicing
   mr_toolkit.trajectory_analysis.traj_analysis

MSM Coarse-graining
-------------------

Coarse-graining
_______________
.. automodule:: mr_toolkit.coarse_graining.msm_coarse_graining
    :members: coarse_grain, compute_avg_bin_weights


Estimating observables
______________________
.. automodule:: mr_toolkit.coarse_graining.msm_coarse_graining
    :members: get_equil, get_comm, get_hill_mfpt, get_naive_hill_mfpt

Constructing matrices
_____________________
.. automodule:: mr_toolkit.coarse_graining.msm_coarse_graining
    :members: build_fine_transition_matrix, build_occupancy


Trajectory Splicing
-------------------

.. automodule:: mr_toolkit.reweighting.splicing
    :members:


Reweighted MSM estimation
--------------------------
.. automodule:: mr_toolkit.reweighting.analysis
    :members:
    :exclude-members: AnalysisRun



AnalysisRun convenience class for MSM estimation
-------------------------------------------------
.. autoclass:: mr_toolkit.reweighting.analysis.AnalysisRun
    :members:
    :undoc-members:
