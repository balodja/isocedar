import numpy as np
import pytest

from coniferest.aadforest import AADForest
from coniferest.datasets import single_outlier
from coniferest.label import Label


def test_scores_negative():
    data = np.arange(1024.0).reshape(256, 4)
    forest = AADForest(n_trees=10, random_seed=0).fit(data)
    scores = forest.score_samples(data)
    assert np.all(scores < 0)


def test_single_outlier():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=0).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_value():
    data, _metadata = single_outlier()
    forest = AADForest(n_trees=10, random_seed=0, prior_influence=2.0).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


def test_prior_influence_callable():
    data, _metadata = single_outlier()
    forest = AADForest(
        n_trees=10,
        random_seed=0,
        prior_influence=lambda ac, nc: np.max(1.0, 0.5 / (ac + nc)),
    ).fit(data)
    scores = forest.score_samples(data)
    # Outlier goes last and must have the lowest score
    assert np.argmin(scores) == data.shape[0] - 1


@pytest.mark.regression
def test_regression_fit_known(regression_data):
    random_seed = 0
    n_samples = 1024
    n_features = 16
    n_known = 16
    n_trees = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([-1, 1], n_known, replace=True)

    forest = AADForest(n_trees=n_trees, random_seed=random_seed)
    forest.fit(data)
    pre_fit_known_scores = forest.score_samples(data)

    forest.fit_known(data, known_data=known_data, known_labels=known_labels)
    scores = forest.score_samples(data)

    assert not np.allclose(pre_fit_known_scores, scores, rtol=1e-3), "Scores must change after fit_known"

    regression_data.assert_allclose(scores)


@pytest.mark.benchmark
@pytest.mark.long
def test_benchmark_fit_known(n_jobs, benchmark):
    benchmark.group = f"AADForest.fit_known {n_jobs = :2d}"
    benchmark.name = "coniferest.aadforest.AADForest"

    random_seed = 0
    n_samples = 1024
    n_features = 16
    n_known = 16
    n_trees = 128
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([-1, 1], n_known, replace=True)

    forest = AADForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    benchmark(forest.fit_known, data, known_data=known_data, known_labels=known_labels)


@pytest.mark.benchmark
@pytest.mark.long
@pytest.mark.parametrize("n_samples", [1 << 10, 1 << 16])
@pytest.mark.parametrize("n_trees", [1 << 10, 1 << 14])
def test_benchmark_loss_gradient(n_samples, n_trees, n_jobs, benchmark):
    benchmark.group = f"AADEvaluator.loss_graident {n_samples = :6d} {n_trees = :4d}, {n_jobs = :2d}"
    benchmark.name = "coniferest.aadforest.AADEvaluator"

    random_seed = 0
    n_features = 16
    n_known = 16
    rng = np.random.default_rng(random_seed)
    data = rng.standard_normal((n_samples, n_features))
    known_data = data[rng.choice(n_samples, n_known, replace=False)]
    known_labels = rng.choice([-1, 1], n_known, replace=True)

    forest = AADForest(n_trees=n_trees, n_jobs=n_jobs, random_seed=random_seed)
    forest.fit(data)

    anomaly_count = np.count_nonzero(known_labels == Label.ANOMALY)
    nominal_count = np.count_nonzero(known_labels == Label.REGULAR)
    scores = forest.score_samples(data)
    q_tau = np.quantile(scores, 1.0 - forest.tau)

    benchmark(
        forest.evaluator.loss_gradient,
        forest.evaluator.weights,
        known_data,
        known_labels,
        anomaly_count,
        nominal_count,
        q_tau,
    )
