import joblib
import numpy as np

from .calc_paths_sum import calc_feature_delta_sum, calc_paths_sum, selector_dtype  # noqa
from .utils import average_path_length

__all__ = ["ForestEvaluator"]


class ForestEvaluator:
    selector_dtype = selector_dtype

    def __init__(self, samples, selectors, indices, leaf_count, *, num_threads):
        """
        Base class for the forest evaluators. Does the trivial job:
        * runs calc_paths_sum written in cython,
        * helps to combine several trees' representations into two arrays.

        Parameters
        ----------
        samples
            Number of samples in every tree.

        selectors
            Array with all the nodes of all the trees.

        indices
            Indices of starting and ending nodes of every tree. For example
            for two trees of length `len1` and `len2` it would be:
            [0, len1, len1 + len2]

        leaf_count
            Total number of leafs in all the trees.

        num_threads : int or None
            Number of threads to use for calculations. If None then
        """
        self.samples = samples

        self.selectors = selectors
        self.indices = indices
        self.leaf_count = leaf_count

        if num_threads is None or num_threads < 1:
            # Count of available CPUs is not a simple thing, see loky's implementation here:
            # https://github.com/joblib/joblib/blob/476ff8e62b221fc5816bad9b55dec8883d4f157c/joblib/externals/loky/backend/context.py#L83
            # We ignore OMP_NUM_THREADS for now
            self.num_threads = joblib.cpu_count()
        else:
            self.num_threads = num_threads

    @classmethod
    def combine_selectors(cls, selectors_list):
        """
        Combine several node arrays into one array of nodes and one array of
        start indices.

        Parameters
        ----------
        selectors_list
            List of node arrays to combine.

        Returns
        -------
        Pair of two arrays: node array and array of starting indices.
        """
        lens = [len(sels) for sels in selectors_list]
        full_len = sum(lens)

        selectors = np.empty((full_len,), dtype=cls.selector_dtype)

        indices = np.empty((len(selectors_list) + 1,), dtype=np.int64)
        indices[0] = 0
        indices[1:] = np.add.accumulate(lens)

        for i in range(len(selectors_list)):
            selectors[indices[i] : indices[i + 1]] = selectors_list[i]

        # Assign a unique sequential index to every leaf
        # The index is used for weighted scores
        leaf_mask = selectors["feature"] < 0
        leaf_count = np.count_nonzero(leaf_mask)
        selectors["left"][leaf_mask] = np.arange(0, leaf_count)

        return selectors, indices, leaf_count

    def score_samples(self, x):
        """
        Perform the computations.

        Parameters
        ----------
        x
            Features to calculate scores of. Should be C-contiguous for performance.

        Returns
        -------
        Array of scores.
        """
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        trees = self.indices.shape[0] - 1

        return -(
            2
            ** (
                -calc_paths_sum(self.selectors, self.indices, x, num_threads=self.num_threads)
                / (self.average_path_length(self.samples) * trees)
            )
        )

    def _feature_delta_sum(self, x):
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        return calc_feature_delta_sum(self.selectors, self.indices, x, num_threads=self.num_threads)

    def feature_signature(self, x):
        delta_sum, hit_count = self._feature_delta_sum(x)

        return delta_sum / hit_count / self.average_path_length(self.samples)

    def feature_importance(self, x):
        delta_sum, hit_count = self._feature_delta_sum(x)

        return np.sum(delta_sum, axis=0) / np.sum(hit_count, axis=0) / self.average_path_length(self.samples)

    @classmethod
    def average_path_length(cls, n_nodes):
        """
        Average path length is abstracted because in different cases we may want to
        use a bit different formulas to make the exact match with other software.

        By default we use our own implementation.
        """
        return average_path_length(n_nodes)
