"""Separate BrainWeb/Rician exploratory benchmark; run from repository root."""
from pathlib import Path
import copy
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path('NGPS').resolve()))
from datasets.manifest import SIGMAS, load_manifest, manifest_hash, resolve_path, stable_seed, validate_manifest
from datasets.preparation import file_hash
from ngps.checkpoint2 import benchmark, freeze, pilot, preflight, read_json, train_matrix
from ngps.provenance import source_record, write_json

ROOT = Path(__file__).resolve().parent
GAUSSIAN = Path('outputs/checkpoint2/medical/mps_full_prepared/manifest.json').resolve()
CONFIG = Path('outputs/checkpoint2/medical/mps_full_run/training.yaml').resolve()


def rician_noise(clean, sigma, seed):
    if not np.isfinite(clean).all() or np.min(clean) < 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Nonnegative finite reference and positive sigma required')
    rng = np.random.default_rng(seed)
    n1 = rng.normal(0, sigma, clean.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, clean.shape).astype(np.float32)
    return np.hypot(clean + n1, n2).astype(np.float32)


def checks():
    # Validate distribution at zero signal (Rayleigh special case), determinism,
    # nonnegativity, finite output and unbounded upper tail; never use medical data.
    zero = np.zeros((100, 100, 100), dtype=np.float32)
    a = rician_noise(zero, .1, 42)
    assert np.array_equal(a, rician_noise(zero, .1, 42))
    assert abs(float(a.mean()) - .1 * np.sqrt(np.pi / 2)) < .0005
    assert abs(float(a.var()) - .01 * (2 - np.pi / 2)) < .00005
    assert np.isfinite(a).all() and a.min() >= 0
    bright = rician_noise(np.ones((10, 30, 30), dtype=np.float32), .1, 42)
    assert bright.max() > 1
    # Independent SciPy Rice distribution's moments at nonzero signal.
    from scipy.stats import rice
    b = rician_noise(np.full(zero.shape, .15, dtype=np.float32), .1, 43)
    mean, var = rice.stats(1.5, scale=.1, moments='mv')
    assert abs(float(b.mean()) - mean) < .0005
    assert abs(float(b.var()) - var) < .00005
    write_json(ROOT / 'noise_tests.json', {'passed': True, 'samples': int(a.size),
               'zero_signal_empirical_mean': float(a.mean()), 'zero_signal_empirical_variance': float(a.var()),
               'nonzero_signal_empirical_mean': float(b.mean()), 'nonzero_signal_theoretical_mean': float(mean),
               'deterministic': True, 'clipped': False})


def main():
    if (ROOT / 'design.json').exists():
        raise FileExistsError('Preserve existing study; use new directory')
    base = load_manifest(GAUSSIAN, prepared=True)
    design = {'kind': 'exploratory_BrainWeb_Rician_matched_to_MPS531',
              'source_tree_sha256': source_record()['tree_sha256'],
              'orchestrator_sha256': file_hash(__file__),
              'parent_gaussian_manifest_hash': manifest_hash(base),
              'formula': 'hypot(x+n1,n2), independent N(0,sigma^2) components; float32, no clipping',
              'normalization': base['normalization'], 'sigmas': list(SIGMAS),
              'noise_seed': 'stable_seed(42,subject,noise:sigma); same n1 stream as Gaussian; subsequent independent n2 stream',
              'training': 'Fresh independent sigma-specific models, full agreed6/2/2 split; seed42, shared width16, one epoch capped531 updates, fixed final checkpoint',
              'methods': ['ngps','blindspot','adjacent'], 'controls': ['nlm','noisy'],
              'controls_limitation': 'NLM uses existing Gaussian-calibrated parameters, not a dedicated Rician correction',
              'budget_seconds': 3600, 'budget_reason': 'Independent Rician MPS pilot and full validation projection before production; fixed531 updates to match completed Gaussian MPS benchmark, not a transferred T4 cap',
              'success': 'NGPS PSNR >= stronger trained comparator at every sigma, zero tolerance',
              'test_subjects': ['subject20','subject44'],
              'limitations': 'Already exposed test subjects: exploratory only, no clean tuning; BrainWeb signal scaling unchanged, not paper IXI normalization/noise levels/training or numerical replication. Equal sigma is component SD, not equal effective Gaussian/Rician MSE.'}
    design['design_hash'] = manifest_hash(design)
    write_json(ROOT / 'design.json', design)
    checks()
    folder = ROOT / 'prepared'
    folder.mkdir(exist_ok=False)
    path = folder / 'manifest.json'
    result = copy.deepcopy(base)
    result['noise'] = 'Synthetic Rician magnitude: hypot(x+n1,n2); independent Gaussian components; no clipping'
    result['rician_study_design_hash'] = design['design_hash']
    result['parent_gaussian_manifest_hash'] = manifest_hash(base)
    for i, row in enumerate(result['subjects']):
        original = base['subjects'][i]
        clean_path = resolve_path(GAUSSIAN, original['clean'])
        if file_hash(clean_path) != original['clean_sha256']:
            raise ValueError('Clean reference hash mismatch during corruption generation')
        clean = np.load(clean_path, allow_pickle=False)
        row['source'] = os.path.relpath(resolve_path(GAUSSIAN, original['source']), folder)
        row['clean'] = os.path.relpath(clean_path, folder)
        subject_folder = folder / row['subject_id']
        subject_folder.mkdir()
        for sigma in SIGMAS:
            key = f'{sigma:.2f}'
            seed = stable_seed(base['seed'], row['subject_id'], f'noise:{key}')
            noisy = rician_noise(clean, sigma, seed)
            dest = subject_folder / f'noisy_{key}.npy'
            np.save(dest, noisy)
            row['noisy'][key] = {'path': str(dest.relative_to(folder)), 'seed': seed,
                                 'sha256': file_hash(dest), 'range': [float(noisy.min()), float(noisy.max())]}
    validate_manifest(result, path)
    write_json(path, result)
    allowed = {str(resolve_path(path, row['noisy'][f'{sigma:.2f}']['path']))
               for row in result['subjects'] if row['split'] in ('train', 'val') for sigma in SIGMAS}
    loads = []
    real_load = np.load
    def guarded_load(p, *args, **kwargs):
        resolved = str(Path(p).resolve())
        if resolved not in allowed:
            raise RuntimeError('Forbidden training array access: ' + resolved)
        loads.append(resolved)
        return real_load(p, *args, **kwargs)
    with patch('numpy.load', side_effect=guarded_load):
        pilot_path = pilot(path, CONFIG, ROOT / 'pilot', device='mps')
        protocol = freeze(path, CONFIG, ROOT / 'experiment', pilot_path,
                          max_train_batches=531, budget_seconds=3600,
                          budget_reason=design['budget_reason'])
        train_matrix(protocol, device='mps')
        preflight(protocol)
    write_json(ROOT / 'training_access_audit.json', {'load_count': len(loads),
               'loaded_paths': sorted(set(loads)), 'forbidden_loads': 0,
               'scope': 'Pilot, training and preflight only; clean files accessed earlier solely to generate corruption, later solely for frozen scoring'})
    results = benchmark(protocol, device='mps')
    print('COMPLETE:', results)


if __name__ == '__main__':
    main()
