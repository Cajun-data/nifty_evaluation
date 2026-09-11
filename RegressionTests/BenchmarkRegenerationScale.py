"""Measure regeneration at a requested synthetic size, without a dense baseline.

Example: python RegressionTests/BenchmarkRegenerationScale.py --samples 10000 --features 5000
Uses independent seeded float64 measurements and balanced labels. This is a
capacity test, not a biological accuracy benchmark. No sampling of pairs is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from BenchmarkMemory import memory, ROOT

sys.path.insert(0, str(ROOT))
from RegeneratingRuleEvaluator import RegeneratingRuleEvaluator


class MeasuredEvaluator(RegeneratingRuleEvaluator):
    def __init__(self, seed):
        super().__init__(seed)
        self.generated_rows = 0
        self.next_report = 100_000
        self.started = time.perf_counter()

    def _vectors(self, ids):
        result = super()._vectors(ids)
        self.generated_rows += len(result)
        if self.generated_rows >= self.next_report:
            print(f'Regenerated {self.generated_rows:,} rule rows; '
                  f'{time.perf_counter() - self.started:.1f} seconds; '
                  f'peak {memory()["peak_working_set_bytes"] / 2**30:.2f} GiB', flush=True)
            self.next_report += 1_000_000
        return result

    def _score_rules(self, labels):
        result = super()._score_rules(labels)
        self.unique_rules = len(result[0])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--features', type=int, default=5000)
    parser.add_argument('--output-dir', default=str(ROOT / 'Test_Output' / 'regeneration_scale'))
    args = parser.parse_args()
    if args.samples < 2 or args.features < 2:
        parser.error('At least two samples and features are required.')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260911)
    # Generate protein-major data to avoid an extra full transpose copy.
    quant = pd.DataFrame(rng.standard_normal((args.features, args.samples)).T,
                         columns=[f'P{i:05}' for i in range(args.features)])
    meta = pd.DataFrame({'classification_label': np.arange(args.samples) % 2})
    config = {'output_dir': str(output), 'k_rules': 15, 'disjoint': False,
              'mutual_information': True, 'mutual_information_cutoff': 0.7}
    evaluator = MeasuredEvaluator(42)
    started = time.perf_counter()
    selected = evaluator.run(config, quant, meta)
    elapsed = time.perf_counter() - started
    assert len(selected) == min(15, evaluator.unique_rules)
    assert selected['Score'].between(0, 1).all()
    assert selected['P_Value'].between(0, 1).all()
    result = dict(memory(), samples=args.samples, features=args.features,
                  pairs=args.features * (args.features - 1) // 2,
                  unique_rules=evaluator.unique_rules, selected_rules=len(selected),
                  pipeline_seconds=elapsed, synthetic_seed=20260911, pipeline_seed=42,
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ('RegeneratingRuleEvaluator.py', 'EvaluateRules.py')},
                  python=sys.version, numpy=np.__version__, pandas=pd.__version__)
    (output / 'measurement.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
