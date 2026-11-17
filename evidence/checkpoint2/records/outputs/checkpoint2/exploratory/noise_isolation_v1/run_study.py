"""Matched Gaussian/Rician component isolation; exploratory, no clean tuning."""
from pathlib import Path
import json
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path('NGPS').resolve()))
from datasets.manifest import load_manifest, manifest_hash, resolve_path
from datasets.preparation import file_hash
from ngps.checkpoint2 import read_json
from ngps.device import synchronize, empty_cache
from ngps.inference import load_model, infer_volume, timed_volume_file
from ngps.metrics import METRICS, aggregate_subjects
from ngps.provenance import provenance, source_record, write_json
from ngps.training import train

ROOT = Path(__file__).resolve().parent
MANIFESTS = {'gaussian': Path('outputs/checkpoint2/medical/mps_full_prepared/manifest.json').resolve(),
             'rician': Path('outputs/checkpoint2/exploratory/rician_v1/prepared/manifest.json').resolve()}
CONFIG = Path('outputs/checkpoint2/medical/mps_full_run/training.yaml').resolve()
VARIANTS = [('full_ngps', 'ngps', None), ('no_rc', 'ngps', 'no_rc'),
            ('no_dynamic', 'ngps', 'no_dynamic'), ('adjacent', 'adjacent', None)]
CAP = 3600


def error_stats(reference, prediction):
    """Descriptive spatial decomposition, NOT stochastic estimator bias/variance."""
    residual = prediction.astype(np.float64) - reference
    result = {}
    for name, mask in [('all', np.ones(reference.shape, dtype=bool)),
                       ('low_x_lt_0.02', reference < .02),
                       ('mid_0.02_le_x_lt_0.10', (reference >= .02) & (reference < .10)),
                       ('high_x_ge_0.10', reference >= .10)]:
        e = residual[mask]
        result[name] = {'pixels': int(e.size), 'mean_signed_error': float(e.mean()) if e.size else None,
                        'mse': float(np.mean(e * e)) if e.size else None,
                        'spatial_centered_error_variance': float(e.var()) if e.size else None}
    return result


def main():
    if (ROOT / 'design.json').exists():
        raise FileExistsError('Preserve existing study')
    manifests = {name: load_manifest(path, prepared=True) for name, path in MANIFESTS.items()}
    # Verify this is a paired noise experiment with identical references/splits.
    for a, b in zip(manifests['gaussian']['subjects'], manifests['rician']['subjects']):
        assert (a['subject_id'], a['split'], a['clean_sha256'], a['shape']) == (b['subject_id'], b['split'], b['clean_sha256'], b['shape'])
    config = yaml.safe_load(CONFIG.read_text())
    assert config['training']['epochs'] == 1
    source_hash = source_record()['tree_sha256']
    design = {'kind': 'exploratory_paired_noise_loss_isolation',
              'orchestrator_sha256': file_hash(__file__), 'source_tree_sha256': source_hash,
              'manifests': {name: {'path': str(MANIFESTS[name]), 'hash': manifest_hash(m)} for name, m in manifests.items()},
              'hardware': provenance('mps'), 'sigma': .06, 'updates': 531, 'config': config,
              'variants': [list(v) for v in VARIANTS], 'budget_seconds_per_run': CAP,
              'budget_reason': 'Fresh per-noise/per-variant MPS pilot before matched531-update production; independent safety cap, no T4 transfer',
              'selection': 'All fixed final checkpoints; no tuning, early stopping or winner selection',
              'comparison': 'PSNR full minus no_dynamic (dynamic contribution conditional on RC); full minus no_rc (RC contribution conditional on dynamic); compare these contrasts across noise models. Not a full RC-by-dynamic factorial.',
              'scoring': 'Whole volume and mean slice PSNR/SSIM/HFEN; equal subject weighting',
              'error_strata': 'Fixed reference-intensity bins x<.02, .02<=x<.10, x>=.10; compute descriptive residual mean, MSE and spatial centered variance only after all models frozen. Not statistical estimator bias/variance.',
              'test_subjects': ['subject20','subject44'], 'seed': 42,
              'limitations': 'Exposed test subjects, single seed/sigma; component removals change objective and cannot identify matching versus target-noise bias alone. Equal sigma is not equal effective corruption. Reference strata are evaluation only.'}
    design['design_hash'] = manifest_hash(design)
    write_json(ROOT / 'design.json', design)
    (ROOT / 'training.yaml').write_text(yaml.safe_dump(config))
    pilots = []
    runs = []
    allowed = {str(resolve_path(MANIFESTS[name], row['noisy']['0.06']['path']))
               for name, m in manifests.items() for row in m['subjects'] if row['split'] in ('train', 'val')}
    real_load = np.load
    loads = []
    def guarded(p, *args, **kwargs):
        resolved = str(Path(p).resolve())
        if resolved not in allowed:
            raise RuntimeError('Forbidden training array load: ' + resolved)
        loads.append(resolved)
        return real_load(p, *args, **kwargs)
    def run(noise, variant, method, ablation, is_pilot, experiment_hash):
        if source_record()['tree_sha256'] != source_hash:
            raise RuntimeError('Source changed after freeze')
        folder = ROOT / ('pilot' if is_pilot else 'runs') / noise / variant
        synchronize('mps')
        start = time.perf_counter()
        ckpt = train(MANIFESTS[noise], ROOT / 'training.yaml', folder,
                     device='mps', sigma=.06, method=method, ablation=ablation,
                     max_train_batches=16 if is_pilot else 531,
                     center_slices=16 if is_pilot else None,
                     train_subjects=['subject04'] if is_pilot else None,
                     max_val_batches=1 if is_pilot else None,
                     gradient_diagnostics=True, experiment_hash=experiment_hash)
        synchronize('mps')
        wall = time.perf_counter() - start
        meta = read_json(ckpt.parent / 'metadata.json')
        row = {'noise': noise, 'variant': variant, 'method': method, 'ablation': ablation,
               'checkpoint': str(ckpt.resolve()), 'checkpoint_sha256': file_hash(ckpt),
               'updates': meta['optimizer_steps'], 'wall_seconds': wall, 'within_cap': wall <= CAP,
               'training_seconds': meta['training_seconds'], 'validation_seconds': meta['validation_seconds'],
               'training_history': meta['training_history'], 'validation_history': meta['validation_history']}
        write_json(folder / 'budget.json', {'wall_seconds': wall, 'cap_seconds': CAP, 'within_cap': wall <= CAP})
        if not row['within_cap'] or row['updates'] != (16 if is_pilot else 531):
            raise RuntimeError('Schedule/budget failed')
        return row
    with patch('numpy.load', side_effect=guarded):
        for noise in MANIFESTS:
            for variant, method, ablation in VARIANTS:
                pilots.append(run(noise, variant, method, ablation, True, design['design_hash']))
                write_json(ROOT / 'pilot.partial.json', pilots)
        projections = [{'noise': p['noise'], 'variant': p['variant'], 'seconds': 1.5 * (
            p['wall_seconds'] - p['training_seconds'] + 358 * p['validation_seconds'] + 531 * p['training_seconds'] / 16)} for p in pilots]
        protocol = {'design_hash': design['design_hash'], 'pilots': pilots, 'projections': projections}
        protocol['protocol_hash'] = manifest_hash(protocol)
        write_json(ROOT / 'protocol.json', protocol)
        if any(p['seconds'] > CAP for p in projections):
            raise RuntimeError('Projection exceeds cap')
        for noise in MANIFESTS:
            for variant, method, ablation in VARIANTS:
                runs.append(run(noise, variant, method, ablation, False, protocol['protocol_hash']))
                write_json(ROOT / 'training.partial.json', runs)
    write_json(ROOT / 'training_access_audit.json', {'load_count': len(loads),
               'loaded_paths': sorted(set(loads)), 'forbidden_loads': 0})
    # Preflight all8 models, complete workloads and test-noisy files before references.
    assert len(runs) == 8
    for r in runs:
        model, meta = load_model(r['checkpoint'], 'cpu', require_selected=True)
        assert file_hash(r['checkpoint']) == r['checkpoint_sha256']
        assert meta['experiment_hash'] == protocol['protocol_hash'] and meta['optimizer_steps'] == 531
        assert meta['manifest_hash'] == manifest_hash(manifests[r['noise']]) and meta['source']['tree_sha256'] == source_hash
        assert meta['config'] == config and meta['sigma'] == .06
        assert meta['method'] == r['method'] and meta.get('exploratory', {}).get('ablation') == r['ablation']
        assert meta['workload'] == {'max_train_batches': 531, 'center_slices': None, 'train_subjects': None, 'max_val_batches': None}
        del model
    for noise, m in manifests.items():
        for row in m['subjects']:
            if row['split'] == 'test':
                assert file_hash(resolve_path(MANIFESTS[noise], row['noisy']['0.06']['path'])) == row['noisy']['0.06']['sha256']
    write_json(ROOT / 'test_started.json', {'protocol_hash': protocol['protocol_hash'], 'runs': runs})
    scores = []
    for noise, manifest in manifests.items():
        for variant in [v[0] for v in VARIANTS] + ['noisy']:
            model = None
            if variant != 'noisy':
                r = next(r for r in runs if r['noise'] == noise and r['variant'] == variant)
                if file_hash(r['checkpoint']) != r['checkpoint_sha256']:
                    raise RuntimeError('Checkpoint changed')
                model, _ = load_model(r['checkpoint'], 'mps', require_selected=True)
                infer_volume(model, np.zeros((1, 256, 256), dtype=np.float32), 'mps')
                function = lambda a: infer_volume(model, a, 'mps')
                device = 'mps'
            else:
                function = lambda a: a.copy()
                device = 'cpu'
            for row in manifest['subjects']:
                if row['split'] != 'test':
                    continue
                clean_path = resolve_path(MANIFESTS[noise], row['clean'])
                assert file_hash(clean_path) == row['clean_sha256']
                prediction, timing = timed_volume_file(function,
                    resolve_path(MANIFESTS[noise], row['noisy']['0.06']['path']),
                    ROOT / 'predictions' / noise / variant / (row['subject_id'] + '.npy'), device)
                ref = np.load(clean_path, allow_pickle=False)
                metrics = {name: float(np.mean([metric(a, b) for a, b in zip(ref, prediction)]))
                           for name, metric in METRICS.items()}
                record = {'noise': noise, 'variant': variant, 'subject': row['subject_id'],
                          **metrics, **timing, 'error_strata': error_stats(ref, prediction)}
                scores.append(record)
                write_json(ROOT / 'scores.partial.json', scores)
            del model
            empty_cache('mps')
    aggregates = []
    for noise in MANIFESTS:
        for variant in [v[0] for v in VARIANTS] + ['noisy']:
            group = [r for r in scores if r['noise'] == noise and r['variant'] == variant]
            aggregates.append({'noise': noise, 'variant': variant,
                               'metrics': aggregate_subjects(group),
                               'error_strata': {region: {key: float(np.mean([r['error_strata'][region][key] for r in group]))
                                    for key in ['mean_signed_error', 'mse', 'spatial_centered_error_variance']}
                                    for region in group[0]['error_strata']}})
    write_json(ROOT / 'summary.json', {'protocol_hash': protocol['protocol_hash'], 'runs': runs,
               'subjects': scores, 'aggregates': aggregates,
               'limitations': design['limitations']})
    print('COMPLETE:', ROOT / 'summary.json')


if __name__ == '__main__':
    main()
