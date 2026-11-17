"""Frozen, matched Checkpoint 2 experiments. No test scoring during pilot or training."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from datasets.manifest import SIGMAS, load_manifest, manifest_hash, noisy_items, resolve_path
from datasets.preparation import file_hash, prepare
from datasets.volume_folder import load_volume
from .baselines import nlm_volume
from .config import load_config
from .device import empty_cache, is_accelerator, require_device, synchronize
from .evaluation import json_metrics
from .guide import guide_filter_volume, make_misalignment_mask
from .inference import infer_volume, load_model, timed_volume_file
from .metrics import METRICS, aggregate_subjects
from .provenance import provenance, source_record, write_json
from .sampler import NeighborGuidedPatchSampler
from .training import train

METHODS = ('ngps', 'blindspot', 'adjacent')
CONTROLS = ('nlm', 'noisy')
SPLITS = {'train': ['subject04', 'subject06', 'subject18', 'subject41', 'subject42', 'subject43'],
          'val': ['subject05', 'subject38'], 'test': ['subject20', 'subject44']}
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare_verified(source_manifest, verification_path, output_root):
    source = load_manifest(source_manifest)
    verification = read_json(verification_path)
    if (verification.get('status') != 'verified_fixed_reference'
            or verification.get('source_manifest_hash') != manifest_hash(source)):
        raise ValueError('Verified source receipt must match the source manifest')
    expected = {r['subject_id']: r['source_sha256'] for r in source['subjects']}
    actual = {r['subject_id']: r['source_sha256'] for r in verification['subjects']}
    if expected != actual:
        raise ValueError('Incomplete source verification')
    if verification.get('normalization', {}).get('policy') != 'already-unit':
        raise ValueError('Unsupported verified normalization policy')
    path = prepare(source_manifest, output_root, 'already-unit')
    manifest = read_json(path)
    manifest['verification'] = verification
    manifest['verification_sha256'] = file_hash(verification_path)
    write_json(path, manifest)
    return path


def pilot(manifest_path, config_path, output_dir, device='cuda'):
    require_device(device)
    manifest = load_manifest(manifest_path, prepared=True)
    if 'subject04' not in {r['subject_id'] for r in manifest['subjects'] if r['split'] == 'train'}:
        raise ValueError('Pilot requires training subject04')
    _, _, config = load_config(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config['training'].update(epochs=1, batch_size=1, seed=42)
    cfg_path = out / 'pilot.yaml'
    cfg_path.write_text(yaml.safe_dump(config))
    measurements = {}
    for method in METHODS:
        start = time.perf_counter()
        checkpoint = train(manifest_path, cfg_path, out / method, device=device, method=method,
                           sigma=0.06, center_slices=16, train_subjects=['subject04'], max_val_batches=1)
        metadata = read_json(checkpoint.parent / 'metadata.json')
        synchronize(device)
        measurements[method] = {key: metadata[key] for key in
                                ('optimizer_steps', 'training_seconds', 'validation_seconds', 'peak_cuda_memory_bytes', 'device_memory')}
        measurements[method]['wall_seconds'] = time.perf_counter() - start
    result = {'schema_version': 1, 'kind': 'pilot', 'manifest_hash': manifest_hash(manifest),
              'source_tree_sha256': source_record()['tree_sha256'],
              'config': config, 'device': str(device), 'measurements': measurements,
              'note': 'Subject04, sigma0.06, 16 center slices, one epoch, batch1. Not held-out evidence.'}
    write_json(out / 'pilot.json', result)
    return out / 'pilot.json'


def freeze(manifest_path, config_path, output_dir, pilot_path=None,
           max_train_batches=None, budget_seconds=600, budget_reason=None, synthetic=False):
    manifest = load_manifest(manifest_path, prepared=True)
    cfg, training, raw = load_config(config_path)
    if training.seed != 42 or cfg.rc_weight != 0.5:
        raise ValueError('Checkpoint2 fixes seed42 and regional-consistency weight0.5')
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('Budget must be finite and positive')
    if budget_seconds > 600 and not budget_reason:
        raise ValueError('A revised budget requires a reason fixed before testing')
    if not synthetic:
        actual = {s: sorted(r['subject_id'] for r in manifest['subjects'] if r['split'] == s) for s in SPLITS}
        if actual != SPLITS:
            raise ValueError('Medical benchmark requires the agreed 6/2/2 subject split')
        if manifest.get('verification', {}).get('status') != 'verified_fixed_reference':
            raise ValueError('Medical preparation requires authoritative verification')
        if pilot_path is None:
            raise ValueError('Run the hosted GPU pilot before freezing a medical experiment')
    tree = source_record()['tree_sha256']
    pilot_record = read_json(pilot_path) if pilot_path else None
    if pilot_record:
        if (pilot_record['manifest_hash'] != manifest_hash(manifest)
                or pilot_record['source_tree_sha256'] != tree):
            raise ValueError('Pilot data or runtime code differs; rerun the pilot')
        pilot_config = pilot_record['config']
        compare = json.loads(json.dumps(raw))
        compare['training'].update(epochs=1, batch_size=1, seed=42)
        if pilot_config != compare or (not synthetic and not is_accelerator(pilot_record['device'])):
            raise ValueError('Pilot must use the same configuration and a CUDA or Apple MPS accelerator')
    if max_train_batches is not None and (type(max_train_batches) is not int or max_train_batches < 1):
        raise ValueError('max_train_batches must be a positive integer')
    estimates = None
    if pilot_record:
        # Conservative projection includes validation/setup overhead; use the
        # slowest objective, identical batch1 pilot and production workload.
        if training.batch_size != 1:
            raise ValueError('Budget projection requires pilot/production batch_size1')
        n_batches = math.ceil(sum(r['shape'][0] - 2 for r in manifest['subjects']
                                  if r['split'] == 'train') / training.batch_size)
        n_val = sum(r['shape'][0] - 2 for r in manifest['subjects'] if r['split'] == 'val')
        step = max(v['training_seconds'] / v['optimizer_steps'] for v in pilot_record['measurements'].values())
        # Full noisy-only validation is measured separately in the pilot below.
        overhead = max(v['wall_seconds'] - v['training_seconds'] for v in pilot_record['measurements'].values())
        validation = max(v['validation_seconds'] for v in pilot_record['measurements'].values())
        fixed = 1.5 * (overhead + training.epochs * validation * n_val)
        affordable = int((budget_seconds - fixed) / (1.5 * training.epochs * max(step, 1e-9)))
        if affordable < 1:
            raise ValueError('Full validation/setup exhausts the budget. Revise common schedule/budget and rerun pilot.')
        if max_train_batches is None:
            max_train_batches = min(n_batches, affordable)
        projected = fixed + 1.5 * training.epochs * min(n_batches, max_train_batches) * step
        if projected > budget_seconds:
            raise ValueError('Requested workload exceeds pilot-based budget; reduce it or justify a larger cap')
        estimates = {'projected_seconds': projected, 'safety_factor': 1.5, 'pilot_sha256': file_hash(pilot_path)}
    elif max_train_batches is None:
        max_train_batches = 1
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_path_out = out / 'training.yaml'
    config_path_out.write_text(yaml.safe_dump(raw))
    figures = {r['subject_id']: {'z': r['shape'][0] // 2, 'y': r['shape'][1] // 2,
                                'x': r['shape'][2] // 2, 'radius': min(32, min(r['shape'][1:]) // 4)}
               for r in manifest['subjects'] if r['split'] == 'test'}
    protocol = {'schema_version': 1, 'kind': 'synthetic' if synthetic else 'medical',
                'upstream_revision': 'c1ab8bcb2dfa80926c7846048b42356546257f8d',
                'manifest': os.path.relpath(Path(manifest_path).resolve(), out),
                'manifest_hash': manifest_hash(manifest), 'source_tree_sha256': tree,
                'training_config': raw, 'config_sha256': file_hash(config_path_out),
                'methods': list(METHODS), 'controls': list(CONTROLS), 'sigmas': list(SIGMAS),
                'seed': 42, 'max_train_batches': max_train_batches,
                'budget_seconds': budget_seconds, 'budget_revision_reason': budget_reason,
                'pilot_projection': estimates, 'figure_coordinates': figures,
                'selection': 'fixed_final_epoch', 'primary_metric': 'PSNR',
                'success': 'At every sigma NGPS mean subject PSNR >= max(blindspot, adjacent); tolerance0dB',
                'tolerance_db': 0.0, 'clipping': False,
                'reference_limitation': manifest.get('verification', {}).get('reference_limitation', 'Synthetic software check only'),
                'loss_convention': 'Course active-pixel normalized static/dynamic reconstruction and RC; RC weight0.5. Paper RC uses an unnormalized sum.',
                'tuning': 'No clean-reference tuning; common configuration and fixed final epoch. No post-test changes to this protocol.'}
    protocol['protocol_hash'] = manifest_hash(protocol)
    write_json(out / 'protocol.json', protocol)
    return out / 'protocol.json'


def load_protocol(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = read_json(protocol_path)
    claimed = protocol.pop('protocol_hash')
    if manifest_hash(protocol) != claimed:
        raise ValueError('Frozen protocol has been modified')
    protocol['protocol_hash'] = claimed
    if protocol['source_tree_sha256'] != source_record()['tree_sha256']:
        raise ValueError('Runtime source changed after freeze; create a separate experiment')
    if file_hash(protocol_path.parent / 'training.yaml') != protocol['config_sha256']:
        raise ValueError('Frozen training configuration changed')
    manifest_path = resolve_path(protocol_path, protocol['manifest'])
    manifest = load_manifest(manifest_path, prepared=True)
    if manifest_hash(manifest) != protocol['manifest_hash']:
        raise ValueError('Prepared manifest changed after freeze')
    return protocol, manifest, manifest_path


def run_name(method, sigma):
    return f'{method}_s{sigma:.2f}'


def validate_checkpoint(metadata, protocol, method, sigma):
    from .inference import validate_selection
    validate_selection(metadata)
    if metadata.get('exploratory', {}).get('ablation') is not None:
        raise ValueError('Ablation checkpoints cannot substitute for full NGPS')
    if (metadata['method'] != method or metadata['sigma'] != sigma
            or metadata['manifest_hash'] != protocol['manifest_hash']
            or metadata.get('experiment_hash') != protocol['protocol_hash']
            or metadata['config'] != protocol['training_config']
            or metadata['seed'] != 42
            or metadata['source']['tree_sha256'] != protocol['source_tree_sha256']
            or metadata.get('workload') != {'max_train_batches': protocol['max_train_batches'],
                'center_slices': None, 'train_subjects': None, 'max_val_batches': None}):
        raise ValueError(f'Checkpoint does not match frozen experiment: {method}/{sigma}')
    per_epoch = min(protocol['max_train_batches'], math.ceil(metadata['training_samples'] /
                    metadata['training_config']['batch_size']))
    if metadata.get('optimizer_steps') != per_epoch * metadata['training_config']['epochs']:
        raise ValueError('Checkpoint did not complete its declared optimizer-step schedule')


def validate_budget(path, protocol):
    record = read_json(path)
    seconds = record.get('wall_seconds')
    if (record.get('cap_seconds') != protocol['budget_seconds'] or record.get('within_cap') is not True
            or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or not 0 < seconds <= protocol['budget_seconds']):
        raise ValueError('Run does not meet declared budget')


def train_matrix(protocol_path, device='cuda'):
    require_device(device)
    protocol, _, manifest_path = load_protocol(protocol_path)
    out = Path(protocol_path).resolve().parent
    if (out / 'test_started.json').exists():
        raise ValueError('Final testing already started; corrective runs require a separate experiment')
    for sigma in SIGMAS:
        for method in METHODS:
            folder = out / 'runs' / run_name(method, sigma)
            checkpoint = folder / 'model_final.pth'
            if checkpoint.exists():
                model, metadata = load_model(checkpoint, 'cpu', require_selected=True)
                validate_checkpoint(metadata, protocol, method, sigma)
                validate_budget(folder / 'budget.json', protocol)
                del model
                continue
            if folder.exists():
                raise FileExistsError(f'Incomplete run preserved at {folder}; use a new experiment directory')
            start = time.perf_counter()
            train(manifest_path, out / 'training.yaml', folder, device=device,
                  sigma=sigma, method=method, max_train_batches=protocol['max_train_batches'],
                  experiment_hash=protocol['protocol_hash'])
            synchronize(device)
            seconds = time.perf_counter() - start
            write_json(folder / 'budget.json', {'wall_seconds': seconds,
                       'cap_seconds': protocol['budget_seconds'], 'within_cap': seconds <= protocol['budget_seconds']})
            if seconds > protocol['budget_seconds']:
                raise RuntimeError('Measured run exceeded declared cap. Preserve evidence; revise budget in a new protocol before testing.')
    return out / 'runs'


def preflight(protocol_path):
    """Validate every run and noisy input before opening any held-out reference."""
    protocol, manifest, manifest_path = load_protocol(protocol_path)
    out = Path(protocol_path).resolve().parent
    checkpoints = {}
    for sigma in SIGMAS:
        for method in METHODS:
            path = out / 'runs' / run_name(method, sigma) / 'model_final.pth'
            if not path.is_file():
                raise ValueError('Missing selected checkpoint: ' + str(path))
            model, metadata = load_model(path, 'cpu', require_selected=True)
            validate_checkpoint(metadata, protocol, method, sigma)
            if (not math.isfinite(metadata['training_history'][-1]['loss'])
                    or len(metadata['training_history']) != metadata['training_config']['epochs']):
                raise ValueError('Incomplete/nonfinite training evidence')
            validate_budget(path.parent / 'budget.json', protocol)
            checkpoints[run_name(method, sigma)] = {'path': path, 'sha256': file_hash(path), 'metadata': metadata}
            del model
        noisy_items(manifest, manifest_path, 'test', sigma)
    return protocol, manifest, manifest_path, checkpoints


def export_diagnostics(manifest_path, config_path, output, subject=None, sigma=0.06, checkpoint=None):
    """One full training-slice triplet, no clean references, portable NPZ arrays."""
    cfg, _, _ = load_config(config_path)
    manifest = load_manifest(manifest_path, prepared=True)
    rows = [r for r in manifest['subjects'] if r['split'] == 'train' and (subject is None or r['subject_id'] == subject)]
    if not rows:
        raise ValueError('Diagnostics require a training subject')
    row = rows[0]
    noisy_items(manifest, manifest_path, 'train', sigma)
    raw = load_volume(resolve_path(manifest_path, row['noisy'][f'{sigma:.2f}']['path']))
    z = raw.shape[0] // 2
    triplet = raw[z - 1:z + 2]
    guides = guide_filter_volume(triplet, cfg)
    tensors = [torch.from_numpy(image)[None, None] for image in guides]
    gc, gp, gn = tensors[1], tensors[0], tensors[2]
    sampler = NeighborGuidedPatchSampler(cfg)
    payload = {'raw_triplet': triplet, 'guide_triplet': guides}
    for direction, guide, image in [('prev', gp, triplet[0]), ('next', gn, triplet[2])]:
        details = sampler.match_details(gc, guide, torch.from_numpy(image)[None, None])
        mask = make_misalignment_mask(gc, guide, cfg.mask_threshold)
        payload[f'{direction}_dynamic_mask'] = mask.numpy()
        payload[f'{direction}_static_mask'] = (1 - mask).numpy()
        payload.update({f'{direction}_{k}': value.numpy() for k, value in details.items()})
    if checkpoint:
        model, metadata = load_model(checkpoint, 'cpu', require_selected=True)
        if metadata['method'] != 'ngps' or metadata['manifest_hash'] != manifest_hash(manifest) or metadata['sigma'] != sigma:
            raise ValueError('Diagnostics checkpoint mismatch')
        payload['prediction_triplet'] = infer_volume(model, triplet, 'cpu')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    write_json(output.with_suffix('.json'), {'subject_id': row['subject_id'], 'z': z, 'sigma': sigma,
               'manifest_hash': manifest_hash(manifest), 'arrays_sha256': file_hash(output),
               'coordinates': 'zero-based y,x in neighboring raw slice; edge coordinates clamped',
               'clean_references_accessed': False})
    diagnostic_figure(output.with_suffix('.png'), payload)
    return output


def diagnostic_figure(output, payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for row, (direction, index) in enumerate((('prev', 0), ('next', 2))):
        arrays = [payload['raw_triplet'][index], payload['guide_triplet'][index],
                  payload[f'{direction}_static_mask'][0, 0],
                  payload[f'{direction}_dynamic_mask'][0, 0],
                  payload[f'{direction}_target'][0, 0], payload[f'{direction}_cost'][0, 0]]
        for column, (name, array) in enumerate(zip(('raw neighbor + matches', 'filtered guide', 'static mask',
                                                   'dynamic mask', 'raw retrieved target', 'mean SSD / area'), arrays)):
            axes[row, column].imshow(array, cmap='magma' if column == 5 else 'gray')
            axes[row, column].set_title(f'{direction}: {name}')
            axes[row, column].axis('off')
        y, x = (size // 2 for size in arrays[0].shape)
        ys = payload[f'{direction}_y'][0, :, y, x]
        xs = payload[f'{direction}_x'][0, :, y, x]
        axes[row, 0].scatter(xs, ys, facecolors='none', edgecolors='red', s=35)
    fig.suptitle('Training-only NGPS diagnostics; red circles: Top-K matches for the central pixel')
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_figure(output, ref, noisy, predictions, coordinate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z, y, x, radius = (coordinate[k] for k in ('z', 'y', 'x', 'radius'))
    names = ['reference', 'noisy', *predictions]
    volumes = [ref, noisy, *predictions.values()]
    fig, axes = plt.subplots(3, len(names), figsize=(3 * len(names), 8), squeeze=False)
    for column, (name, volume) in enumerate(zip(names, volumes)):
        image = volume[z]
        axes[0, column].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0, column].set_title(name)
        axes[1, column].imshow(image[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1],
                              cmap='gray', vmin=0, vmax=1)
        axes[2, column].imshow(np.abs(image - ref[z]), cmap='magma', vmin=0, vmax=0.2)
        for axis in axes[:, column]:
            axis.axis('off')
    fig.suptitle('Frozen coordinates: full slice / central detail / absolute reference error (0–0.2)')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def benchmark(protocol_path, device='cuda'):
    require_device(device)
    protocol, manifest, manifest_path, checkpoints = preflight(protocol_path)
    out = Path(protocol_path).resolve().parent
    results = out / 'results'
    results.mkdir(exist_ok=False)
    write_json(out / 'test_started.json', {'protocol_hash': protocol['protocol_hash'],
               'checkpoint_sha256': {k:v['sha256'] for k,v in checkpoints.items()},
               'note': 'Testing has started. Failed/partial attempts must be reported; do not tune this experiment.'})
    rows = []
    slice_rows = []
    test_rows = [r for r in manifest['subjects'] if r['split'] == 'test']
    # Warmup uses zeros shaped from manifest metadata, never a clean reference.
    for sigma in SIGMAS:
        for method in METHODS + CONTROLS:
            model = None
            if method in METHODS:
                selected = checkpoints[run_name(method, sigma)]
                if file_hash(selected['path']) != selected['sha256']:
                    raise ValueError('Selected checkpoint changed after preflight')
                model, _ = load_model(selected['path'], device, require_selected=True)
                for shape in {tuple(row['shape'][1:]) for row in test_rows}:
                    infer_volume(model, np.zeros((1, *shape), dtype=np.float32), device)
                predict = lambda volume: infer_volume(model, volume, device)
                execution_device = device
            elif method == 'nlm':
                predict = lambda volume: nlm_volume(volume, sigma)
                execution_device = 'cpu'
                nlm_volume(np.zeros((1, *test_rows[0]['shape'][1:]), dtype=np.float32), sigma)
            else:
                predict = lambda volume: volume.copy()
                execution_device = 'cpu'
            hardware = provenance(execution_device)
            for row in test_rows:
                noisy_path = resolve_path(manifest_path, row['noisy'][f'{sigma:.2f}']['path'])
                clean_path = resolve_path(manifest_path, row['clean'])
                # No clean arrays are opened until all nine runs passed preflight.
                if file_hash(clean_path) != row['clean_sha256']:
                    raise ValueError('Held-out reference hash mismatch')
                destination = results / 'predictions' / run_name(method, sigma) / (row['subject_id'] + '.npy')
                prediction, times = timed_volume_file(predict, noisy_path, destination, execution_device)
                ref = load_volume(clean_path, unit_range=True)
                if list(prediction.shape) != row['shape']:
                    raise ValueError('Prediction does not cover the complete volume')
                scores = [{name: metric(r, p) for name, metric in METRICS.items()}
                          for r, p in zip(ref, prediction)]
                slice_rows.extend({'subject_id': row['subject_id'], 'sigma': sigma, 'method': method,
                                   'z': z, **score} for z, score in enumerate(scores))
                record = {'subject_id': row['subject_id'], 'sigma': sigma, 'method': method,
                          'slices': len(prediction), 'device': str(execution_device),
                          'device_name': hardware['device_name'], **times,
                          **{name: float(np.mean([score[name] for score in scores])) for name in METRICS},
                          'prediction_sha256': file_hash(destination),
                          'checkpoint_sha256': checkpoints[run_name(method, sigma)]['sha256'] if method in METHODS else None}
                rows.append(record)
                write_json(results / 'subjects.partial.json', json_metrics(rows))
            del model
            empty_cache(device)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['sigma'])].append(row)
    aggregates = []
    for (method, sigma), group in grouped.items():
        for metric, stats in aggregate_subjects(group).items():
            aggregates.append({'method': method, 'sigma': sigma, 'metric': metric, **stats})
        for metric in ('inference_seconds', 'volume_seconds'):
            values = [r[metric] for r in group]
            aggregates.append({'method': method, 'sigma': sigma, 'metric': metric,
                               'mean': float(np.mean(values)), 'std_population': float(np.std(values)),
                               'n_subjects': len(values)})
    success = []
    for sigma in SIGMAS:
        means = {r['method']: r['mean'] for r in aggregates if r['sigma'] == sigma and r['metric'] == 'PSNR'}
        strongest = max(('blindspot', 'adjacent'), key=means.get)
        gap = 0.0 if means['ngps'] == means[strongest] else means['ngps'] - means[strongest]
        success.append({'sigma': sigma, 'strongest_comparator': strongest, 'ngps_psnr': means['ngps'],
                        'comparator_psnr': means[strongest], 'gap_db': gap, 'tolerance_db': 0.0,
                        'target_met': means['ngps'] >= means[strongest]})
    figures = results / 'figures'
    figures.mkdir()
    for row in test_rows:
        ref = load_volume(resolve_path(manifest_path, row['clean']), unit_range=True)
        for sigma in SIGMAS:
            noisy = load_volume(resolve_path(manifest_path, row['noisy'][f'{sigma:.2f}']['path']))
            predictions = {method: load_volume(results / 'predictions' / run_name(method, sigma) / (row['subject_id'] + '.npy'))
                           for method in METHODS + ('nlm',)}
            comparison_figure(figures / f"{row['subject_id']}_s{sigma:.2f}.png", ref, noisy, predictions,
                              protocol['figure_coordinates'][row['subject_id']])
    report = {'protocol_hash': protocol['protocol_hash'], 'kind': protocol['kind'], 'subjects': rows,
              'aggregates': aggregates, 'success_by_sigma': success, 'all_targets_met': all(r['target_met'] for r in success),
              'reference_limitation': protocol['reference_limitation'],
              'variability': 'Two independent held-out subjects; one training seed. Population SD, no clinical inference.',
              'timing': 'Warmup/model loading excluded. volume_seconds starts before load and ends after save; metrics/hash checks excluded.',
              'hardware': provenance(device), 'checkpoint_sha256': {k:v['sha256'] for k,v in checkpoints.items()}}
    write_json(results / 'benchmark.json', json_metrics(report))
    _write_csv(results / 'subjects.csv', rows)
    _write_csv(results / 'slices.csv', slice_rows)
    failures = []
    for (method, sigma), group in grouped.items():
        for subject in (r['subject_id'] for r in group):
            worst = min((r for r in slice_rows if r['method'] == method and r['sigma'] == sigma
                         and r['subject_id'] == subject), key=lambda r: r['PSNR'])
            failures.append({**worst, 'selection': 'posthoc minimum slice PSNR; descriptive failure analysis only'})
    _write_csv(results / 'worst_slices.csv', failures)
    _write_csv(results / 'aggregates.csv', aggregates)
    _write_csv(results / 'success.csv', success)
    write_json(results / 'COMPLETE.json', {'protocol_hash': protocol['protocol_hash'],
               'benchmark_sha256': file_hash(results / 'benchmark.json'), 'subject_rows': len(rows)})
    return results
