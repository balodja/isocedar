import numpy as np

from .evaluator import ForestEvaluator
from .utils import average_path_length

# Not very useful classes at the moment.
# Implemented purely with the educational purpose.


class RandomLimeForest:
    def __init__(self, trees=100, subsamples=256, depth=None, seed=0):
        self.subsamples = subsamples
        self.trees = trees
        self.depth = depth

        self.seedseq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(seed)

        self.estimators = []
        self.n = 0

    def fit(self, data):
        n = data.shape[0]
        self.n = n
        self.subsamples = self.subsamples if n > self.subsamples else n

        self.depth = self.depth or int(np.ceil(np.log2(self.subsamples)))

        self.estimators = [None] * self.trees
        seeds = self.seedseq.spawn(self.trees)
        for i in range(self.trees):
            subs = self.rng.choice(n, self.subsamples)
            gen = RandomLimeGenerator(data[subs, :], self.depth, seeds[i])
            self.estimators[i] = gen.pine

        return self

    def mean_paths(self, data):
        means = np.zeros(data.shape[0])
        for ti in range(self.trees):
            path = self.estimators[ti].paths(data)
            means += path

        means /= self.trees
        return means

    def scores(self, data):
        means = self.mean_paths(data)
        return -(2 ** (-means / average_path_length(self.subsamples)))


class RandomLime:
    def __init__(self, features, selectors, values):
        self.features = features
        self.len = selectors.shape[0]

        # Two complementary arrays.
        # Selectors select feature to branch on.
        self.selectors = selectors
        # Values either set the deciding feature value or set the closing path length
        self.values = values

    def _get_one_path(self, key):
        i = 1
        while 2 * i < self.selectors.shape[0]:
            f = self.selectors[i]
            if f < 0:
                break

            if key[f] <= self.values[i]:
                i = 2 * i
            else:
                i = 2 * i + 1

        return self.values[i]

    def paths(self, x):
        n = x.shape[0]
        paths = np.empty(n)
        for i in range(n):
            paths[i] = self._get_one_path(x[i, :])

        return paths


class RandomLimeGenerator:
    def __init__(self, sample, depth, seed=0):
        self.depth = depth
        self.features = sample.shape[1]
        self.length = 1 << (depth + 1)
        self.rng = np.random.default_rng(seed)
        self.selectors = np.full(self.length, -1, dtype=np.int32)
        self.values = np.full(self.length, 0, dtype=np.float64)

        self._populate(1, sample)

        self.pine = RandomLime(self.features, self.selectors, self.values)

    def _populate(self, i, sample):
        if sample.shape[0] == 1:
            self.values[i] = np.floor(np.log2(i))
            return

        if self.length <= 2 * i:
            self.values[i] = np.floor(np.log2(i)) + average_path_length(sample.shape[0])

            return

        selector = self.rng.integers(self.features)
        self.selectors[i] = selector

        minval = np.min(sample[:, selector])
        maxval = np.max(sample[:, selector])
        if minval == maxval:
            self.selectors[i] = -1
            self.values[i] = np.floor(np.log2(i)) + average_path_length(sample.shape[0])

            return

        value = self.rng.uniform(minval, maxval)
        self.values[i] = value

        self._populate(2 * i, sample[sample[:, selector] <= value])
        self._populate(2 * i + 1, sample[sample[:, selector] > value])


class LimeEvaluator(ForestEvaluator):
    def __init__(self, pine_forest):
        pines = pine_forest.estimators
        self.trees = len(pines)
        if self.trees < 1:
            raise ValueError("a forest without trees?")

        selectors, node_offsets, leaf_offsets = self.combine_selectors([self.extract_selectors(pine) for pine in pines])

        super(LimeEvaluator, self).__init__(
            samples=pine_forest.subsamples, selectors=selectors, node_offsets=node_offsets, leaf_offsets=leaf_offsets
        )

    @classmethod
    def extract_selectors(cls, pine):
        selectors = np.zeros((pine.len,), dtype=cls.selector_dtype)

        mapping = np.full((pine.len,), -1, dtype=np.int32)
        current = 0
        for i in range(1, pine.len):
            if pine.selectors[i] != -1 or pine.values[i] != 0:
                mapping[i] = current
                current += 1

        selectors = selectors[:current]

        for i in range(1, pine.len):
            if pine.selectors[i] != -1 or pine.values[i] != 0:
                current = mapping[i]

                feature = pine.selectors[i]
                selectors[current]["feature"] = feature
                selectors[current]["value"] = pine.values[i]

                if 2 * i >= pine.len:
                    continue

                selectors[current]["left"] = mapping[2 * i]
                selectors[current]["right"] = mapping[2 * i + 1]

        return selectors
