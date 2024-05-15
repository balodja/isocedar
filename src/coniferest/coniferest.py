from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from sklearn.ensemble._bagging import _generate_indices  # noqa
from sklearn.tree._criterion import MSE  # noqa
from sklearn.tree._splitter import RandomSplitter  # noqa
from sklearn.tree._tree import DepthFirstTreeBuilder, Tree  # noqa
from sklearn.utils.validation import check_random_state

from .evaluator import ForestEvaluator
from .utils import average_path_length

__all__ = ["Coniferest", "ConiferestEvaluator"]

# Instead of doing:
# from sklearn.utils._random import RAND_R_MAX
# we have:
RAND_R_MAX = 0x7FFFFFFF


# Cause RAND_R_MAX is restricted to C-code.


class Coniferest(ABC):
    """
    Base class for the forests in the package. It settles the basic
    low-level machinery with the sklearn's trees, used here.

    Parameters
    ----------
    trees : list or None, optional
        List with the trees in the forest. If None, then empty list is used.

    n_subsamples : int, optional
        Subsamples to use for the training.

    max_depth : int or None, optional
        Maximum depth of the trees in use. If None, then `log2(n_subsamples)`

    n_jobs : int, optional
        Number of threads to use for scoring. If -1, then number of CPUs is used.

    random_seed : int or None, optional
        Seed for the reproducibility. If None, then random seed is used.
    """

    def __init__(self, trees=None, n_subsamples=256, max_depth=None, n_jobs=-1, random_seed=None):
        self.trees = trees or []
        self.n_subsamples = n_subsamples
        self.max_depth = max_depth or int(np.log2(n_subsamples))

        self.n_jobs = n_jobs

        # For the better future with reproducible parallel tree building.
        # self.seedseq = np.random.SeedSequence(random_state)
        # rng, = self.seedseq.spawn(1)
        # self.rng = np.random.default_rng(rng)

        self.rng = np.random.default_rng(random_seed)

        # The following are the setting for the tree building procedures.

        # May we use the same data points during subsampling? No.
        self.bootstrap_samples = False
        # How many samples the node should have at least to perform a split? Two.
        self.min_samples_split = 2
        # How many samples the leaf might have? One.
        self.min_samples_leaf = 1
        # Don't know what it is.
        self.min_weight_leaf = 0
        # Don't know. Deprecated. And deleted in newer version of sklearn.
        self.min_impurity_decrease = 0
        # How many outputs does each experiment (point) have? Zero can't be in sklearn.
        self.n_outputs = 1

    def build_trees(self, data, n_trees):
        """
        Just build `n_trees` trees from supplied `data`.

        Parameters
        ----------
        data
            Features.

        n_trees
            Number of trees to build

        Returns
        -------
        List of trees.
        """
        n_population, n_features = data.shape

        n_samples = self.n_subsamples
        if n_samples > n_population:
            msg1 = "population should be greater or equal than subsamples number"
            msg2 = f"got n_population < n_subsamples ({n_population} < {n_samples})"
            msg3 = f"assuming n_subsamples = {n_population}"
            warn(msg1 + ", " + msg2 + ", " + msg3)
            n_samples = n_population

        trees = []
        for tree_index in range(n_trees):
            random_state = check_random_state(self.rng.integers(RAND_R_MAX))
            indices = _generate_indices(
                random_state=random_state,
                bootstrap=self.bootstrap_samples,
                n_population=n_population,
                n_samples=n_samples,
            )

            subsamples = data[indices, :]
            tree = self.build_one_tree(subsamples)
            trees.append(tree)

        return trees

    def build_one_tree(self, data):
        """
        Build just one tree.

        Parameters
        ----------
        data
            Features to build that one tree of.

        Returns
        -------
        A tree.
        """
        # Hollow plug
        criterion = MSE(self.n_outputs, self.n_subsamples)

        # Splitter for splitting the nodes.
        splitter_state = check_random_state(self.rng.integers(RAND_R_MAX))
        splitter = RandomSplitter(
            criterion=criterion,
            max_features=1,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_leaf=self.min_weight_leaf,
            random_state=splitter_state,
            monotonic_cst=None,
        )

        builder_args = {
            "splitter": splitter,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_leaf": self.min_weight_leaf,
            "max_depth": self.max_depth,
            "min_impurity_decrease": self.min_impurity_decrease,
        }

        # Initialize the builder
        builder = DepthFirstTreeBuilder(**builder_args)

        # Initialize the tree
        n_samples, n_features = data.shape
        tree = Tree(n_features, np.array([1] * self.n_outputs, dtype=np.int64), self.n_outputs)

        # Cause of sklearn bugs we cannot do this:
        # y = np.zeros((n_samples, self.n_outputs))
        # Instead we do:
        y = np.empty((n_samples, self.n_outputs))
        y_column = np.arange(n_samples)
        for oi in range(self.n_outputs):
            y[:, oi] = y_column
        # The counterpart is rnd.uniform from sklearn.ensemble.IsolationForest.fit.

        # And finally build that tree.
        builder.build(tree, data, y)

        return tree

    @abstractmethod
    def fit(self, data, labels=None):
        """
        Fit to the applied data.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_known(self, data, known_data=None, known_labels=None):
        """
        Fit to the applied data with priors.
        """
        raise NotImplementedError()

    @abstractmethod
    def score_samples(self, samples):
        """
        Evaluate scores for samples.
        """
        raise NotImplementedError()

    @abstractmethod
    def feature_signature(self, x):
        raise NotImplementedError()

    @abstractmethod
    def feature_importance(self, x):
        raise NotImplementedError()


class ConiferestEvaluator(ForestEvaluator):
    """
    Fast evaluator of scores for Coniferests.

    Parameters
    ----------
    coniferest : Coniferest
        The forest for building the evaluator from.
    map_value : callable or None
        Optional function to map leaf values, mast accept 1-D array of values
        and return an array of the same shape.
    """

    def __init__(self, coniferest, map_value=None):
        selectors_list = [self.extract_selectors(t, map_value) for t in coniferest.trees]
        selectors, node_offsets, leaf_offsets = self.combine_selectors(selectors_list)

        super().__init__(
            samples=coniferest.n_subsamples,
            selectors=selectors,
            node_offsets=node_offsets,
            leaf_offsets=leaf_offsets,
            num_threads=coniferest.n_jobs,
        )

    @classmethod
    def extract_selectors(cls, tree, map_value=None):
        """
        Extract node representations for the tree.

        Parameters
        ----------
        tree
            Tree to extract selectors from.
        map_value
            Optional function to map leaf values

        Returns
        -------
        Array with selectors.
        """
        nodes = tree.__getstate__()["nodes"]
        selectors = np.zeros_like(nodes, dtype=cls.selector_dtype)

        selectors["feature"] = nodes["feature"]
        selectors["feature"][selectors["feature"] < 0] = -1

        selectors["left"] = nodes["left_child"]
        selectors["right"] = nodes["right_child"]
        selectors["value"] = nodes["threshold"]

        n_node_samples = nodes["n_node_samples"]

        selectors["log_n_node_samples"] = np.log(n_node_samples)

        def correct_values(i, depth):
            if selectors[i]["feature"] < 0:
                value = depth + average_path_length(n_node_samples[i])
                selectors[i]["value"] = value if map_value is None else map_value(value)
            else:
                correct_values(selectors[i]["left"], depth + 1)
                correct_values(selectors[i]["right"], depth + 1)

        correct_values(0, 0)

        return selectors
