"""Exact compatibility checks for bounded pair generation and scoring."""
import unittest
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pandas as pd

from DataTransformer import DataTransformer
from EvaluateRules import EvaluateRules


class MemoryOptimizationTests(unittest.TestCase):
    def test_pair_blocks_preserve_missing_values_ties_and_order(self):
        quant = pd.DataFrame({
            'a': [np.nan, np.nan, 1., np.inf, -np.inf, 2.],
            'b': [np.nan, 1., np.nan, np.inf, -np.inf, 2.],
            'c': [3., 0., 1., -np.inf, np.nan, 2.],
            'd': [0., 0., 0., 0., 0., 0.],
        })
        original = quant.copy(deep=True)
        pairs = list(combinations(quant.columns, 2))
        values = quant.fillna(-np.inf).to_numpy()
        indices = list(combinations(range(quant.shape[1]), 2))
        expected = (values[:, [p[0] for p in indices]].T >
                    values[:, [p[1] for p in indices]].T).astype(np.int8)
        for elements in (1, 12, 30, 10_000):
            with self.subTest(elements=elements), patch.object(DataTransformer, '_PAIR_BLOCK_ELEMENTS', elements):
                actual = DataTransformer().vectorize_all_pairs(pairs, quant)
                np.testing.assert_array_equal(actual, expected, strict=True)
                self.assertTrue(actual.flags.c_contiguous)
                evaluator = EvaluateRules()
                actual_unique, actual_pairs = evaluator.remove_identical_rules_structured_view(actual, pairs)
                expected_unique, expected_pairs = evaluator.remove_identical_rules_structured_view(expected, pairs)
                np.testing.assert_array_equal(actual_unique, expected_unique, strict=True)
                self.assertEqual(actual_pairs, expected_pairs)
        pd.testing.assert_frame_equal(quant, original, check_exact=True)
        self.assertEqual(DataTransformer().vectorize_all_pairs([], quant).shape, (0, 6))
        self.assertEqual(DataTransformer().vectorize_all_pairs(pairs, quant.iloc[:0]).shape, (6, 0))

    def test_scoring_is_exact_across_blocks_and_does_not_overflow(self):
        rng = np.random.default_rng(14)
        matrix = rng.integers(0, 2, size=(17, 513), dtype=np.int8)
        matrix[0] = 1
        for labels in (np.arange(513) % 2, np.ones(513, dtype=int), np.zeros(513, dtype=int)):
            evaluator = EvaluateRules(seed=42)
            evaluator._n_pos = np.sum(labels == 1)
            evaluator._n_neg = np.sum(labels == 0)
            def original_score(y):
                tp = matrix.dot(y) / evaluator._n_pos if evaluator._n_pos else 0
                fp = matrix.dot(1 - y) / evaluator._n_neg if evaluator._n_neg else 0
                return np.abs(tp - fp)
            for elements in (1, 1026, 5000, 100_000):
                with self.subTest(elements=elements), patch.object(EvaluateRules, '_PAIR_BLOCK_ELEMENTS', elements):
                    np.testing.assert_array_equal(evaluator.evaluate_pairs(matrix, labels), original_score(labels), strict=True)
                    seed = evaluator.seed
                    expected = original_score(np.random.default_rng(seed).permutation(labels))
                    np.testing.assert_array_equal(evaluator.get_null_scores(matrix, labels), expected, strict=True)
                    self.assertEqual(evaluator.seed, seed + 1)


if __name__ == '__main__':
    unittest.main()
