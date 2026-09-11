"""Exact feature selection without storing the pair-by-sample rule matrix."""
import os
import sys
from hashlib import md5

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from Colors import Colors
from EvaluateRules import EvaluateRules


class RegeneratingRuleEvaluator(EvaluateRules):
    def _vectors(self, ids):
        pairs = self._pairs[ids]
        # Consecutive combination blocks usually share their first protein.
        if len(pairs) and np.all(pairs[:, 0] == pairs[0, 0]):
            left = self._values[pairs[0, 0]]
        else:
            left = self._values[pairs[:, 0]]
        return (left > self._values[pairs[:, 1]]).astype(np.int8)

    def _score_vectors(self, vectors, labels):
        # Vectors and labels are binary. Counting gives exactly the same integer
        # dot products without promoting the complete block to wide integers.
        positive = np.count_nonzero(vectors[:, labels == 1], axis=1)
        negative = np.count_nonzero(vectors, axis=1) - positive
        tp = positive / self._n_pos if self._n_pos > 0 else 0
        fp = negative / self._n_neg if self._n_neg > 0 else 0
        return np.abs(tp - fp)

    def _fingerprint(self, packed_row):
        # A digest only locates candidates. Equality is always verified below.
        return md5(packed_row.tobytes()).digest()

    def _prepare(self, quant_df):
        self._proteins = quant_df.columns.to_numpy()
        # Contiguous protein rows allow efficient regeneration from sample columns.
        self._values = np.ascontiguousarray(quant_df.fillna(-np.inf).to_numpy().T)
        n = len(self._proteins)
        self._pairs = np.empty((n * (n - 1) // 2, 2), dtype=np.uint32)
        start = 0
        for first in range(n - 1):
            end = start + n - first - 1
            self._pairs[start:end, 0] = first
            self._pairs[start:end, 1] = np.arange(first + 1, n, dtype=np.uint32)
            start = end
        self._block_size = max(1, self._PAIR_BLOCK_ELEMENTS // max(1, self._values.shape[1]))

    def _unique_ids(self, fingerprints):
        """Keep the first original pair for each exact binary vector, even on collisions."""
        order = np.argsort(fingerprints, kind='stable')
        sorted_fingerprints = fingerprints[order]
        same = np.flatnonzero(sorted_fingerprints[1:] == sorted_fingerprints[:-1])
        del sorted_fingerprints
        keep = np.ones(len(fingerprints), dtype=bool)
        if len(same):
            splits = np.flatnonzero(np.diff(same) > 1) + 1
            starts = np.r_[same[0], same[splits]]
            ends = np.r_[same[splits - 1] + 2, same[-1] + 2]
            for start, end in zip(starts, ends):
                group = order[start:end]
                anchor = self._vectors(group[:1])[0]
                # Store IDs, never an unbounded collection of regenerated vectors.
                other_representatives = []
                for offset in range(1, len(group), self._block_size):
                    ids = group[offset:offset + self._block_size]
                    vectors = self._vectors(ids)
                    equal = np.all(vectors == anchor, axis=1)
                    keep[ids[equal]] = False
                    for position in np.flatnonzero(~equal):
                        pair_id = ids[position]
                        for representative in other_representatives:
                            if np.array_equal(vectors[position], self._vectors([representative])[0]):
                                keep[pair_id] = False
                                break
                        else:
                            other_representatives.append(pair_id)
        return np.flatnonzero(keep)

    def _score_rules(self, labels):
        count = len(self._pairs)
        fingerprints = np.empty(count, dtype='V16')
        true_scores = np.empty(count, dtype=np.float64)
        buckets = np.empty(count, dtype=np.uint8)
        print(' - FINGERPRINTING AND SCORING RULE BLOCKS', file=sys.stderr, flush=True)
        for start in range(0, count, self._block_size):
            stop = min(start + self._block_size, count)
            vectors = self._vectors(slice(start, stop))
            true_scores[start:stop] = self._score_vectors(vectors, labels)
            buckets[start:stop] = self.get_proportion_bucket_list(vectors)
            packed = np.packbits(vectors, axis=1)
            fingerprints[start:stop] = [self._fingerprint(row) for row in packed]
        print(' - VERIFYING DUPLICATES BY REGENERATION', file=sys.stderr, flush=True)
        ids = self._unique_ids(fingerprints)
        del fingerprints
        true_scores = true_scores[ids]
        buckets = buckets[ids]
        print(f'{Colors.INFO}INFO: {len(ids)} rules remaining after filtering out identical rules.{Colors.END}',
              file=sys.stderr, flush=True)

        print(' - REGENERATING RULES FOR NULL SCORES', file=sys.stderr, flush=True)
        # Exactly one initial permutation, followed by the existing small-bucket sequence.
        shuffled = self.randomize_labels(labels)
        null_scores = np.empty(len(ids), dtype=np.float64)
        for start in range(0, len(ids), self._block_size):
            stop = start + self._block_size
            null_scores[start:stop] = self._score_vectors(self._vectors(ids[start:stop]), shuffled)

        # The legacy summary groups buckets in first-occurrence order, not numeric order.
        unique_buckets, first = np.unique(buckets, return_index=True)
        bucket_order = unique_buckets[np.argsort(first)]
        bucket_rank = np.empty(101, dtype=np.uint8)
        p_values = np.empty(len(ids), dtype=np.float64)
        for rank, bucket in enumerate(bucket_order):
            bucket_rank[bucket] = rank
            positions = np.flatnonzero(buckets == bucket)
            null_distribution = null_scores[positions]
            if len(positions) < 100:
                scores = list(null_distribution)
                while len(scores) < 100:
                    shuffled = self.randomize_labels(labels)
                    # Regenerate at most one block even for small buckets with many samples.
                    for start in range(0, len(positions), self._block_size):
                        block_ids = ids[positions[start:start + self._block_size]]
                        scores.extend(self._score_vectors(self._vectors(block_ids), shuffled))
                null_distribution = np.asarray(scores)
            null_distribution = np.sort(null_distribution)
            index = np.searchsorted(null_distribution, true_scores[positions], side='left')
            p_values[positions] = (len(null_distribution) - index) / len(null_distribution)
        return ids, true_scores, buckets, p_values, bucket_rank

    def _frame(self, positions, ids, scores, buckets, p_values):
        # Called for k selected rows, or explicitly requested diagnostic output only.
        return pd.DataFrame({
            'Gene_Pair': [tuple(self._proteins[self._pairs[ids[p]]]) for p in positions],
            'True_Score': scores[positions],
            'Bucket': buckets[positions].astype(int),
            'P_Value': p_values[positions],
        })

    def _select(self, configs, ids, scores, buckets, p_values, bucket_rank):
        # Preserve the stable legacy sort: p-value, descending score, then summary order.
        ranked = np.lexsort((ids, bucket_rank[buckets], -scores, p_values))
        selected = []
        used_proteins = set()
        used_rules = set()
        cached = {}
        mutual_info = configs['mutual_information']
        disjoint = configs['disjoint']
        k = configs['k_rules']
        for position in ranked:
            rule = tuple(self._proteins[self._pairs[ids[position]]])
            if disjoint and any(protein in used_proteins for protein in rule):
                continue
            if mutual_info:
                vector = self._vectors([ids[position]])[0]
                if not used_rules:
                    used_rules.add(rule)
                    used_proteins.update(rule)
                    cached[rule] = vector
                    selected.append(position)
                    # Match the original first-rule control flow, including k=1.
                    continue
                if any(normalized_mutual_info_score(vector, cached[kept]) >= configs['mutual_information_cutoff']
                       for kept in used_rules):
                    if len(selected) >= k:
                        break
                    continue
                used_rules.add(rule)
                cached[rule] = vector
            used_proteins.update(rule)
            selected.append(position)
            if len(selected) >= k:
                break
        if len(selected) < k:
            if disjoint and mutual_info:
                description = 'disjoint pairs with low mutual information'
            elif mutual_info:
                description = 'pairs with low mutual information'
            elif disjoint:
                description = 'disjoint pairs'
            else:
                description = None
            if description:
                print(f'{Colors.WARNING}WARNING: Only {len(selected)} {description} available (requested {k}).{Colors.END}',
                      file=sys.stderr, flush=True)
        return self._frame(np.asarray(selected, dtype=np.intp), ids, scores, buckets, p_values)

    def run(self, configs, quant_df, meta_df, *, return_details=False):
        """Return selected features; full legacy-shaped diagnostics are opt-in."""
        labels = self.binarize_labels(meta_df)
        self._prepare(quant_df)
        try:
            print(f'{Colors.INFO}INFO: {len(self._pairs)} rules generated from {len(self._proteins)} proteins.{Colors.END}',
                  file=sys.stderr, flush=True)
            ids, scores, buckets, p_values, bucket_rank = self._score_rules(labels)
            print('FILTERING RULES', file=sys.stderr, flush=True)
            selected = self._select(configs, ids, scores, buckets, p_values, bucket_rank)
            self.save_rules(selected, os.path.join(configs['output_dir'], 'selected_features.tsv'))
            if return_details:
                summary_order = np.lexsort((ids, bucket_rank[buckets]))
                summary = self._frame(summary_order, ids, scores, buckets, p_values)
                return scores, summary, selected
            return selected
        finally:
            # Do not retain quantification or pair arrays on an evaluator kept by a caller.
            del self._values, self._pairs, self._proteins
