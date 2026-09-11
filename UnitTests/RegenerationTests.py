"""Exact dense/regenerated comparisons, including deliberate fingerprint collisions."""
from contextlib import redirect_stderr
from io import StringIO
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from EvaluateRules import EvaluateRules
from FeatureSelector import FeatureSelector
from RegeneratingRuleEvaluator import RegeneratingRuleEvaluator


class RegenerationTests(unittest.TestCase):
    def compare(self, quant, labels, *, seed=42, k=5, disjoint=False, mi=True,
                cutoff=0.7, collision=False, elements=33):
        meta = pd.DataFrame({'classification_label': labels})
        original = quant.copy(deep=True)
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            config = {'output_dir': directory, 'k_rules': k, 'disjoint': disjoint,
                      'mutual_information': mi, 'mutual_information_cutoff': cutoff}
            dense = EvaluateRules(seed)
            np.random.seed(71)
            expected = dense.run_rule_evaluator(config, list(combinations(quant.columns, 2)), quant, meta)
            expected_bytes = (Path(directory) / 'selected_features.tsv').read_bytes()
            expected_rng = np.random.get_state()
            regenerated = RegeneratingRuleEvaluator(seed)
            patches = {'_PAIR_BLOCK_ELEMENTS': elements}
            if collision:
                patches['_fingerprint'] = lambda self, row: b'\0' * 16
            np.random.seed(71)
            with patch.multiple(RegeneratingRuleEvaluator, **patches):
                actual = regenerated.run(config, quant, meta, return_details=True)
            np.testing.assert_array_equal(actual[0], expected[0], strict=True)
            for a, e in zip(actual[1:], expected[1:]):
                pd.testing.assert_frame_equal(a, e, check_exact=True)
            self.assertEqual((Path(directory) / 'selected_features.tsv').read_bytes(), expected_bytes)
            self.assertEqual(regenerated.seed, dense.seed)
            actual_rng = np.random.get_state()
            self.assertEqual(actual_rng[0], expected_rng[0])
            np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
            self.assertEqual(actual_rng[2:], expected_rng[2:])
            self.assertFalse(hasattr(regenerated, '_values'))
        pd.testing.assert_frame_equal(quant, original, check_exact=True)

    def test_all_filters_ties_collisions_and_small_null_buckets(self):
        rng = np.random.default_rng(12)
        quant = pd.DataFrame(rng.integers(-2, 4, (31, 8)).astype(float), columns=list('abcdefgh'))
        quant.iloc[::3, 0] = np.nan
        quant['h'] = quant['g']
        quant.iloc[0, 2:4] = [np.inf, -np.inf]
        for disjoint, mi, k, collision in product((False, True), (False, True), (1, 5), (False, True)):
            with self.subTest(disjoint=disjoint, mi=mi, k=k, collision=collision):
                self.compare(quant, np.arange(31) % 2, disjoint=disjoint, mi=mi, k=k, collision=collision)

    def test_unseeded_rng_sequence_and_large_counts(self):
        rng = np.random.default_rng(17)
        quant = pd.DataFrame(rng.normal(size=(513, 9)), columns=list('abcdefghi'))
        self.compare(quant, np.arange(513) % 2, seed=None, elements=1026)

    def test_all_duplicate_rules(self):
        quant = pd.DataFrame(np.ones((17, 12)), columns=[str(i) for i in range(12)])
        self.compare(quant, np.arange(17) % 2, collision=True)

    def test_count_scores_match_integer_dot_products(self):
        rng = np.random.default_rng(32)
        vectors = rng.integers(0, 2, (11, 2001), dtype=np.int8)
        vectors[0] = 1
        for labels in (np.arange(2001) % 2, np.zeros(2001, dtype=int), np.ones(2001, dtype=int)):
            evaluator = RegeneratingRuleEvaluator(42)
            evaluator._n_pos = np.sum(labels == 1)
            evaluator._n_neg = np.sum(labels == 0)
            np.testing.assert_array_equal(evaluator._score_vectors(vectors, labels),
                                          evaluator.evaluate_pairs(vectors, labels), strict=True)

    def test_bucket_counts_above_expansion_threshold(self):
        rng = np.random.default_rng(28)
        quant = pd.DataFrame(rng.normal(size=(50, 40)), columns=[str(i) for i in range(40)])
        for elements in (51, 1000, 1_000_000):
            self.compare(quant, np.arange(50) % 2, elements=elements)

    def test_selected_only_path_avoids_dense_evaluator(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            config = {'output_dir': directory, 'k_rules': 2, 'disjoint': False,
                      'mutual_information': False, 'mutual_information_cutoff': 0.7, 'seed': 42,
                      'filtered_feature_quant_table': pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 1]}),
                      'feature_meta_table': pd.DataFrame({'classification_label': [0, 1, 0]})}
            with patch.object(EvaluateRules, 'run_rule_evaluator', side_effect=AssertionError('Dense path used')):
                selected = FeatureSelector().find_features(config, return_details=False)
            self.assertIsInstance(selected, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
