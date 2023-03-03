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
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.coarse_grain
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.compute_avg_bin_weights


Estimating observables
______________________
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.get_equil
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.get_comm
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.get_hill_mfpt
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.get_naive_hill_mfpt

Constructing matrices
_____________________
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.build_fine_transition_matrix
.. automethod:: mr_toolkit.coarse_graining.msm_coarse_graining.build_occupancy


Trajectory Splicing
-------------------

.. automethod:: mr_toolkit.reweighting.splicing.get_receiving_distribution

.. automethod:: mr_toolkit.reweighting.splicing.splice_trajectory
.. automethod:: mr_toolkit.reweighting.splicing.splice_trajectories
.. automethod:: mr_toolkit.reweighting.splicing.iterative_trajectory_splicing

Reweighted MSM estimation
--------------------------
.. automethod:: mr_toolkit.reweighting.analysis.compute_reweighted_stationary
.. automethod:: mr_toolkit.reweighting.analysis.get_kl


AnalysisRun convenience class for MSM estimation
-------------------------------------------------
.. autoclass:: mr_toolkit.reweighting.analysis.AnalysisRun
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_avg_kl

.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_stationary
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_stationary_naive
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_pyemma_stationary
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_reweighted_stationary

.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.iterative_trajectory_splicing
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.splice_trajectories

.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.compute_mfpt
.. automethod:: mr_toolkit.reweighting.analysis.AnalysisRun.mfpt