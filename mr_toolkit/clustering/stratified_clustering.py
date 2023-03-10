import numpy as np
from sklearn.cluster import KMeans
from numpy.typing import ArrayLike
import tqdm.auto as tqdm
from copy import deepcopy


class StratifiedClusters:
    """Class for performing stratified k-means clustering."""

    def __init__(self, n_clusters: int, bin_bounds: ArrayLike):
        """

        Parameters
        ----------
        n_clusters: int, Number of clusters in each stratum

        bin_bounds: array-like, boundaries of stratified bins. Should not include -inf, +inf
        """

        self.n_clusters = n_clusters

        self.bin_boundaries = np.concatenate([[-np.inf], bin_bounds, [np.inf]])

        self.kmeans_models = {}

        self.kmeans_seed = 1337
        self.max_iter = 1000

        self.coord_to_stratify = None

        self.disable_progress = False

    def fit(self, data: ArrayLike, coord_to_stratify: int = 0):
        """
        Fits the stratified clusterer model.

        Parameters
        ----------
        data: Input points. Should be 2 dimensions, (frame, coordinates).

        coord_to_stratify: int, Coordinate to stratify on (i.e. traject

        Todo
        -----
        Instead of providing a coord to stratify, provide a separate set of
        trajectories to stratify on. The length must match the input data.
        This can just be one dimension of the input data... Or something else!
        """

        if self.coord_to_stratify is not None and not self.coord_to_stratify == coord_to_stratify:
            print(f"Warning: Changing the coordinate to stratify from {self.coord_to_stratify} to {coord_to_stratify}")
        self.coord_to_stratify = coord_to_stratify

        assert len(np.array(data).shape) <= 2, "Dimensionality not correct, expected ndim<=2"

        for i, (bin_lower, bin_upper) in tqdm.tqdm(
                enumerate(zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])),
                total=len(self.bin_boundaries) - 1):

            # print(f"=== Processing bin {i}, from {bin_lower} - {bin_upper}")

            kmeans_estimator = KMeans(
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                n_init='auto'
            )

            # Get the points in this bin
            points_in_bin = np.where(
                (data[..., self.coord_to_stratify] >= bin_lower) &
                (data[..., self.coord_to_stratify] < bin_upper)
            )

            try:
                kmeans_estimator.fit(data[points_in_bin])
            except ValueError as e:
                print(i, bin_lower, bin_upper)
                print(points_in_bin)
                raise e

            self.kmeans_models[i] = deepcopy(kmeans_estimator)

    def predict(self, data: ArrayLike):
        """
        Assigns stratified clusters to a set of input data.

        Parameters
        ----------
        data: Array-like, The set of samples to assign to clusters

        Returns
        -------
        Integer cluster assignments
        """

        discretized = np.full((data.shape[0]), fill_value=-1, dtype=int)

        cluster_offset = 0

        for i, (bin_lower, bin_upper) in tqdm.tqdm(
                enumerate(zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])),
                total=len(self.bin_boundaries) - 1,
                disable=self.disable_progress):

            # Get the points in this bin
            points_in_bin = np.where(
                (data[:, self.coord_to_stratify] >= bin_lower) &
                (data[:, self.coord_to_stratify] < bin_upper)
            )

            _clustering = self.kmeans_models[i]

            # If no matches, skip (duh)
            if not points_in_bin[0].shape == (0,):

                discretization = _clustering.predict(data[points_in_bin])
                discretized[points_in_bin] = discretization
                discretized[points_in_bin] = discretized[points_in_bin] + cluster_offset

            cluster_offset += len(_clustering.cluster_centers_)

        assert not -1 in discretized, "Something didn't get correctly discretized"
        return discretized

    def remove_state(self, state_to_remove: int):
        """
        Removes a cluster by index, and re-indexes the remaining clusters to be consecutive.

        Parameters
        ----------
        state_to_remove: int, The index of the state to remove

        Returns
        -------
        The index of the removed state, in the space of the ORIGINAL clustering the model was built with.
        """

        cluster_offset = 0

        for i, bin_bounds in enumerate(zip(self.bin_boundaries[:-1], self.bin_boundaries[1:])):

            _clustering = self.kmeans_models[i]

            # Check if any of the states to be removed are in this bin
            if state_to_remove in range(cluster_offset, cluster_offset + len(_clustering.cluster_centers_)):

                index_within_stratum = state_to_remove - cluster_offset
                # print(f"Index of state {state_to_remove} within stratum {i} is {index_within_stratum}
                # (offset {cluster_offset})")
                _clustering.cluster_centers_ = np.delete(_clustering.cluster_centers_, index_within_stratum, axis=0)

                # Get the original index, before any cleaning was done
                original_index = index_within_stratum + i * self.n_clusters
                return original_index

            cluster_offset += len(_clustering.cluster_centers_)

    @property
    def cluster_centers(self):

        cluster_centers = []

        for model in self.kmeans_models.values():
            cluster_centers.append(model.cluster_centers_)

        return np.concatenate(cluster_centers)

# # Use this to make a test, uh, later
# strat = StratifiedClusters(1, [1, 5, 10])
# # Test it out with 8x 2-dimensional trajectories
# # With 2 clusters these should be, roughly: [2,3,4,5,6,7,0,1]
# # With 1, they should be [1,1,2,2,3,3,0,0]
# test_points = np.array([[3.3, 4.2], [4.3, 1.0], [7.7, 9.2], [9.5, 14.7], [105, 300], [200.3, 900.3], [-100, -102], [-12.3, -56]])
# strat.fit(test_points, 0)
# strat.predict(test_points)
