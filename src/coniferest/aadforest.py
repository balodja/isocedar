from numbers import Real
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .calc_paths_sum import calc_paths_sum, calc_paths_sum_transpose  # noqa
from .coniferest import Coniferest, ConiferestEvaluator
from .label import Label

__all__ = ["AADForest"]


class AADEvaluator(ConiferestEvaluator):
    def __init__(self, aad):
        super(AADEvaluator, self).__init__(aad, map_value=lambda x: -np.reciprocal(x))
        self.weights = np.full(shape=(self.n_leaves,), fill_value=np.reciprocal(np.sqrt(self.n_leaves)))

    def score_samples(self, x, weights=None):
        """
        Perform the computations.

        Parameters
        ----------
        x
            Features to calculate scores of. Should be C-contiguous for performance.
        weights
            Specific leaf weights to use instead of self.weights

        Returns
        -------
        Array of scores.
        """
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        if weights is None:
            weights = self.weights

        return calc_paths_sum(
            self.selectors,
            self.node_offsets,
            x,
            weights,
            num_threads=self.num_threads,
            batch_size=self.get_batch_size(self.n_trees),
        )

    def loss(
        self,
        weights,
        known_data,
        known_labels,
        anomaly_count,
        nominal_count,
        q_tau,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        """Loss for the known data.

        Adopted from Eq3 of Das et al. 2019 https://arxiv.org/abs/1901.08930
        with the respect to
        1) different anomaly labels - they use +1 for anomalies, -1 for
              nomalies, we use `Label` enum, which is opposite and includes
              `UNKNOWN` label;
        2) different direction of the score axis - they use higher scores
              for anomalies, we use lower scores for anomalies.
        """

        # This new score is negative for the "anomalous" subsample and
        # positive for the "nominal" subsample.
        scores = self.score_samples(known_data, weights) - q_tau

        if prior_weights is None:
            prior_weights = self.weights

        loss = 0.0
        if anomaly_count:
            # For anomalies in "nominal" subsample we add their positive scores.
            loss += C_a * np.sum(scores[(known_labels == Label.ANOMALY) & (scores >= 0)]) / anomaly_count
        if nominal_count:
            # Add for nominals in "anomalous" subsample we add their inverse scores (positive number).
            loss -= np.sum(scores[(known_labels == Label.REGULAR) & (scores <= 0)]) / nominal_count
        delta_weights = weights - prior_weights
        loss += 0.5 * prior_influence * np.inner(delta_weights, delta_weights)

        return loss

    def loss_gradient(
        self,
        weights,
        known_data,
        known_labels,
        anomaly_count,
        nominal_count,
        q_tau,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        scores = self.score_samples(known_data, weights) - q_tau

        if prior_weights is None:
            prior_weights = self.weights

        sample_weights = np.zeros(known_data.shape[0])
        if anomaly_count:
            sample_weights[(known_labels == Label.ANOMALY) & (scores >= 0)] = C_a / anomaly_count
        if nominal_count:
            sample_weights[(known_labels == Label.REGULAR) & (scores <= 0)] = -1.0 / nominal_count

        grad = calc_paths_sum_transpose(
            self.selectors,
            self.node_offsets,
            self.leaf_offsets,
            known_data,
            sample_weights,
            num_threads=self.num_threads,
            batch_size=self.get_batch_size(len(known_data)),
        )
        delta_weights = weights - prior_weights
        grad += prior_influence * delta_weights

        return grad

    def loss_hessian(
        self,
        weights,
        vector,
        known_data,
        known_labels,
        q_tau,
        anomaly_count,
        nominal_count,
        C_a=1.0,
        prior_influence=1.0,
        prior_weights=None,
    ):
        return vector * prior_influence


class AADForest(Coniferest):
    """
    Active Anomaly Detection with Isolation Forest.

    See Das et al., 2017 https://arxiv.org/abs/1708.09441

    Parameters
    ----------
    n_trees : int, optional
        Number of trees in the isolation forest.

    n_subsamples : int, optional
        How many subsamples should be used to build every tree.

    max_depth : int or None, optional
        Maximum depth of every tree. If None, `log2(n_subsamples)` is used.

    n_jobs : int or None, optional
        Number of threads to use for scoring. If None - all available CPUs are used.

    random_seed : int or None, optional
        Random seed to use for reproducibility. If None - random seed is used.

    prior_influence : float or callable, optional
        An regularization coefficient value in the loss functioin. Default is 1.0.
        Signature: '(anomaly_count, nominal_count) -> float'
    """

    def __init__(
        self,
        n_trees=100,
        n_subsamples=256,
        max_depth=None,
        tau=0.97,
        C_a=1.0,
        prior_influence=1.0,
        n_jobs=None,
        random_seed=None,
        sampletrees_per_batch=1 << 20,
    ):
        super().__init__(
            trees=[],
            n_subsamples=n_subsamples,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_seed=random_seed,
            sampletrees_per_batch=sampletrees_per_batch,
        )
        self.n_trees = n_trees
        self.tau = tau
        self.C_a = C_a

        if isinstance(prior_influence, Callable):
            self.prior_influence = prior_influence
        elif isinstance(prior_influence, Real):
            self.prior_influence = lambda anomaly_count, nominal_count: prior_influence
        else:
            raise ValueError("prior_influence is neither a callable nor a constant")

        self.evaluator = None

    def _build_trees(self, data):
        if len(self.trees) == 0:
            self.trees = self.build_trees(data, self.n_trees)
            self.evaluator = AADEvaluator(self)

    def fit(self, data, labels=None):
        """
        Build the trees with the data `data`.

        Parameters
        ----------
        data
            Array with feature values of objects.

        labels
            Optional. Labels of objects. May be regular, anomalous or unknown.
            See `Label` data for details.

        Returns
        -------
        self
        """
        # If no labels were supplied, train with them.
        if labels is None:
            return self.fit_known(data)

        # Otherwise select known data, and train on it.
        labels = np.asarray(labels)
        index = labels != Label.UNKNOWN
        return self.fit_known(data, data[index, :], labels[index])

    def fit_known(self, data, known_data=None, known_labels=None):
        """
        The same `fit` but with a bit of different API. Known data and labels
        are separated from training data for time and space optimality. High
        chances are that `known_data` is much smaller that `data`. At that case
        it is not reasonable to hold the labels for whole `data`.

        Parameters
        ----------
        data
            Training data (array with feature values) to build trees with.

        known_data
            Feature values of known data.

        known_labels
            Labels of known data.

        Returns
        -------
        self
        """

        known_data = np.asarray(known_data) if known_data is not None else None
        known_labels = np.asarray(known_labels) if known_labels is not None else None

        self._build_trees(data)

        if (
            known_data is None
            or len(known_data) == 0
            or known_labels is None
            or len(known_labels) == 0
            or np.all(known_labels == Label.UNKNOWN)
        ):
            return self

        scores = self.score_samples(data)
        # Our scores are negative, so we need to "invert" the quantile.
        q_tau = np.quantile(scores, 1.0 - self.tau)

        anomaly_count = np.count_nonzero(known_labels == Label.ANOMALY)
        nominal_count = np.count_nonzero(known_labels == Label.REGULAR)
        prior_influence = self.prior_influence(anomaly_count, nominal_count)

        def fun(weights):
            return self.evaluator.loss(
                weights, known_data, known_labels, anomaly_count, nominal_count, q_tau, self.C_a, prior_influence
            )

        def jac(weights):
            return self.evaluator.loss_gradient(
                weights, known_data, known_labels, anomaly_count, nominal_count, q_tau, self.C_a, prior_influence
            )

        def hessp(weights, vector):
            return self.evaluator.loss_hessian(
                weights,
                vector,
                known_data,
                known_labels,
                anomaly_count,
                nominal_count,
                q_tau,
                self.C_a,
                prior_influence,
            )

        res = minimize(fun, self.evaluator.weights, method="trust-krylov", jac=jac, hessp=hessp, tol=1e-4)
        weights_norm = np.sqrt(np.inner(res.x, res.x))
        self.evaluator.weights = res.x / weights_norm

        return self

    def score_samples(self, samples):
        """
        Computer scores for the supplied data.

        Parameters
        ----------
        samples
            Feature values to compute scores on.

        Returns
        -------
        Array with computed scores.
        """
        return self.evaluator.score_samples(samples)

    def feature_signature(self, x):
        raise NotImplementedError()

    def feature_importance(self, x):
        raise NotImplementedError()
