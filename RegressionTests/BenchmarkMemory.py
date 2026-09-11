"""Windows before/after benchmark using the included regression data.

Run with Python >=3.11 and the project dependencies installed:
    python RegressionTests/BenchmarkMemory.py --baseline-ref da3db09 --repeats 3
Raw outputs, source snapshots, logs and measurements go under Test_Output.
Each measurement uses a fresh process and Windows' peak working-set counter.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Test_Output' / 'ram_benchmark'


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_working_set_bytes': counters.PeakWorkingSetSize,
            'working_set_bytes': counters.WorkingSetSize,
            'peak_commit_bytes': counters.PeakPagefileUsage}


def worker(source, config, destination):
    sys.path.insert(0, str(source))
    import nifty
    import numpy
    import pandas
    import sklearn
    import scipy
    import statsmodels
    import cloudpickle
    # Capture stage-boundary memory without retaining any pipeline objects.
    stages = {}
    original_train = nifty.ModelGenerator.run_model_generator
    def train(self, configs):
        stages['before_training'] = memory()
        stages['retained_keys_before_training'] = sorted(configs)
        return original_train(self, configs)
    nifty.ModelGenerator.run_model_generator = train
    sys.argv = ['nifty.py', '-c', str(config)]
    started = time.perf_counter()
    nifty.main()
    result = memory()
    result['pipeline_seconds'] = time.perf_counter() - started
    result['stages'] = stages
    result['python'] = sys.version
    result['versions'] = {m.__name__: m.__version__ for m in (numpy, pandas, sklearn, scipy, statsmodels, cloudpickle)}
    Path(destination).write_text(json.dumps(result, indent=2), encoding='utf-8')


def configured(template, changes):
    text = template.read_text(encoding='utf-8')
    for key, value in changes.items():
        literal = json.dumps(value)
        text, count = re.subn(r'^' + re.escape(key) + r'\s*=.*$', key + ' = ' + literal, text, flags=re.M)
        if count != 1:
            raise ValueError(f'Expected exactly one {key} in {template}')
    return text


def run(baseline_ref, repeats):
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = OUT / 'baseline'
    commit = subprocess.check_output(['git', 'rev-parse', baseline_ref], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=ROOT, text=True).splitlines()
    for name in files:
        if name.endswith('.py'):
            target = baseline / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT))
    # Exercise the reference-splitting path using the three disjoint fixture sets.
    for kind in ('quant', 'meta'):
        frames = [pd.read_csv(ROOT / f'RegressionTests/test_data/{part}_{kind}.tsv', sep='\t')
                  for part in ('FS', 'Train_Test', 'Val')]
        combined = pd.concat(frames, ignore_index=True)
        assert combined.sample_id.is_unique
        combined.to_csv(OUT / f'reference_{kind}.tsv', sep='\t', index=False)
    cases = {
        'features': ('FeatureSelectionTest', {}),
        'features_train': ('ModelGeneratorTest', {}),
        'full': ('ApplyClassifierTest', {}),
        'train_only': ('ModelGenOnlyTest', {}),
        'apply_only': ('ApplyClassifierOnlyTest', {}),
        'full_probabilities': ('ApplyClassifierTest', {'prediction_format': 'probabilities'}),
        'reference_full': ('ApplyClassifierTest', {
            'input_files': 'reference',
            'reference_quant_file': (OUT / 'reference_quant.tsv').as_posix(),
            'reference_meta_file': (OUT / 'reference_meta.tsv').as_posix()}),
        'svm_probabilities': ('ApplyClassifierTest', {'model_type': 'SVM', 'prediction_format': 'probabilities'}),
    }
    records = []
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONIOENCODING='utf-8')
    for case, (fixture, overrides) in cases.items():
        for repeat in range(repeats if case in ('features', 'full', 'reference_full') else 1):
            # Alternate version order to reduce systematic warm-cache bias.
            versions = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
            for version in versions:
                destination = OUT / case / str(repeat + 1) / version
                destination.mkdir(parents=True, exist_ok=True)
                config = destination / 'config.toml'
                config.write_text(configured(ROOT / 'RegressionTests' / fixture / 'config.toml',
                                  dict(overrides, output_dir=destination.as_posix())), encoding='utf-8')
                source = baseline if version == 'before' else ROOT
                started = time.perf_counter()
                with (destination / 'run.log').open('w', encoding='utf-8') as log:
                    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                                str(source), str(config), str(destination / 'memory.json')],
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{case}/{version} failed: {destination / "run.log"}')
                measurement = json.loads((destination / 'memory.json').read_text())
                if version == 'after' and 'retained_keys_before_training' in measurement['stages']:
                    keys = set(measurement['stages']['retained_keys_before_training'])
                    assert 'feature_table' in keys
                    assert not keys.intersection({
                        'rules', 'true_scores', 'all_evaluated_rules', 'feature_quant_table',
                        'filtered_feature_quant_table', 'feature_meta_table',
                        'reference_quant_table', 'reference_meta_table'})
                records.append(dict(case=case, repeat=repeat + 1, version=version,
                                    wall_seconds=time.perf_counter() - started, **measurement))
                print(f'{case} {repeat + 1} {version}: {measurement["peak_working_set_bytes"] / 2**20:.1f} MiB, '
                      f'{measurement["pipeline_seconds"]:.2f} s', flush=True)
                (OUT / 'measurements.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
            before = OUT / case / str(repeat + 1) / 'before'
            after = OUT / case / str(repeat + 1) / 'after'
            hashes = {}
            for name in ('selected_features.tsv', 'model_information.txt', 'predicted_classes.tsv',
                         'trained_model_and_model_metadata.pkl'):
                if (before / name).exists():
                    a, b = (before / name).read_bytes(), (after / name).read_bytes()
                    if a != b:
                        raise AssertionError(f'Output differs: {case}/{name}')
                    hashes[name] = hashlib.sha256(a).hexdigest()
            (OUT / case / str(repeat + 1) / 'exact_comparison.json').write_text(json.dumps(hashes, indent=2))
            print(f'  EXACT MATCH: {", ".join(hashes)}', flush=True)
    manifest = {'baseline_commit': commit,
                'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'RegressionTests' / 'test_data').glob('*.tsv'))},
                'optimized_source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                           for name in ('DataTransformer.py', 'EvaluateRules.py', 'nifty.py')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='da3db09')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', nargs=3, metavar=('SOURCE', 'CONFIG', 'RESULT'))
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('This benchmark uses Windows process memory counters.')
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.worker:
        worker(*args.worker)
    else:
        run(args.baseline_ref, args.repeats)
