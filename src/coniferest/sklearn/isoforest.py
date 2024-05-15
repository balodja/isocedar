import numpy as np

from ..evaluator import ForestEvaluator


class IsolationForestEvaluator(ForestEvaluator):
    def __init__(self, isoforest):
        """
        Create evaluator for sklearn's version of isolation forest.

        Parameters
        ----------
        isoforest
            Sklearn's isolation forest instance.
        """
        selectors_list = [self.extract_selectors(e) for e in isoforest.estimators_]
        selectors, node_offsets, leaf_offsets = self.combine_selectors(selectors_list)

        super(IsolationForestEvaluator, self).__init__(
            samples=isoforest.max_samples_,
            selectors=selectors,
            node_offsets=node_offsets,
            leaf_offsets=leaf_offsets,
            num_threads=isoforest.n_jobs,
        )

    @classmethod
    def extract_selectors(cls, estimator):
        nodes = estimator.tree_.__getstate__()["nodes"]
        selectors = np.zeros_like(nodes, dtype=cls.selector_dtype)

        selectors["feature"] = nodes["feature"]
        selectors["feature"][selectors["feature"] < 0] = -1

        selectors["left"] = nodes["left_child"]
        selectors["right"] = nodes["right_child"]
        selectors["value"] = nodes["threshold"]

        n_node_samples = nodes["n_node_samples"]

        def correct_values(i, depth):
            if selectors[i]["feature"] < 0:
                selectors[i]["value"] = depth + cls.average_path_length(n_node_samples[i])
            else:
                correct_values(selectors[i]["left"], depth + 1)
                correct_values(selectors[i]["right"], depth + 1)

        correct_values(0, 0)

        return selectors

    @classmethod
    def average_path_length(cls, n):
        """
        Our average_path_length is a bit different from sklearn's one.
        So we reproduce the sklearn's realization here.
        """
        if n <= 1:
            return 0
        elif n == 2:
            return 1
        else:
            return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n
