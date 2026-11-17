"""Frozen full-workload noisy-only loss ablations; run from repository root."""
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path('NGPS').resolve()))
from datasets.manifest import load_manifest, manifest_hash
from ngps.checkpoint2 import read_json
from ngps.device import synchronize
from ngps.provenance import provenance, source_record, write_json
from ngps.training import train

ROOT = Path(__file__).resolve().parent
MANIFEST = Path('outputs/checkpoint2/medical/mps_full_prepared/manifest.json').resolve()
BASE = Path('outputs/checkpoint2/medical/mps_full_run/training.yaml').resolve()
VARIANTS = [('full_ngps', 'ngps', None), ('ngps_no_rc', 'ngps', 'no_rc'),
            ('ngps_no_dynamic', 'ngps', 'no_dynamic'), ('adjacent', 'adjacent', None)]
CAP = 3600


def main():
    if (ROOT / 'design.json').exists():
        raise FileExistsError('Preserve existing study; use a new directory')
    manifest = load_manifest(MANIFEST, prepared=True)
    config = yaml.safe_load(BASE.read_text())
    config['training']['epochs'] = 2
    batches = sum(r['shape'][0] - 2 for r in manifest['subjects'] if r['split'] == 'train')
    assert batches == 1074 and config['training']['batch_size'] == 1
    source = source_record()
    design = {'kind': 'exploratory_noisy_only_full_workload_loss_ablation',
              'manifest': str(MANIFEST), 'manifest_hash': manifest_hash(manifest),
              'source_tree_sha256': source['tree_sha256'], 'hardware': provenance('mps'),
              'variants': [list(v) for v in VARIANTS], 'config': config,
              'sigma': 0.06, 'epochs': 2, 'updates': 2148, 'batches_per_epoch': 1074,
              'validation': 'All358 interior noisy validation slices; fixed masked monitoring only',
              'normalization': 'Original active-pixel means. Remove designated term without rescaling remaining terms.',
              'ablation_timing': 'Matching and neighbor forwards still computed; not optimized speed comparisons',
              'selection': 'Fixed final epoch, all variants reported, no model/hyperparameter winner selected',
              'budget_seconds_per_run': CAP,
              'budget_reason': 'Independent MPS two-full-epoch ablation study; new per-variant pilot and safety-factor projection before production, not inherited T4 timing',
              'pilot': 'Each variant: subject04,16center slices,one epoch,one noisy validation batch',
              'test_scoring': False, 'clean_reference_selection': False,
              'limitations': 'Single seed/sigma, no clean risk estimates or quality ranking; previously exposed test subjects remain closed.'}
    design['design_hash'] = manifest_hash(design)
    write_json(ROOT / 'design.json', design)
    config_path = ROOT / 'training.yaml'
    config_path.write_text(yaml.safe_dump(config))
    pilot_config = {**config, 'training': {**config['training'], 'epochs': 1}}
    pilot_path = ROOT / 'pilot.yaml'
    pilot_path.write_text(yaml.safe_dump(pilot_config))
    allowed = {str((MANIFEST.parent / r['noisy']['0.06']['path']).resolve())
               for r in manifest['subjects'] if r['split'] in ('train', 'val')}
    loads = []
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        if resolved not in allowed:
            raise RuntimeError('Forbidden array access: ' + resolved)
        loads.append(resolved)
        return original_load(path, *args, **kwargs)
    def run(name, method, ablation, pilot=False, experiment_hash=None):
        if source_record()['tree_sha256'] != source['tree_sha256']:
            raise RuntimeError('Source changed after design freeze')
        folder = ROOT / ('pilot' if pilot else 'runs') / name
        synchronize('mps')
        start = time.perf_counter()
        checkpoint = train(MANIFEST, pilot_path if pilot else config_path, folder,
                           device='mps', sigma=0.06, method=method, ablation=ablation,
                           gradient_diagnostics=True, max_train_batches=16 if pilot else 1074,
                           center_slices=16 if pilot else None,
                           train_subjects=['subject04'] if pilot else None,
                           max_val_batches=1 if pilot else None,
                           experiment_hash=experiment_hash)
        synchronize('mps')
        elapsed = time.perf_counter() - start
        meta = read_json(checkpoint.parent / 'metadata.json')
        row = {'variant': name, 'method': method, 'ablation': ablation,
               'updates': meta['optimizer_steps'], 'wall_seconds': elapsed,
               'within_cap': elapsed <= CAP, 'training_history': meta['training_history'],
               'validation_history': meta['validation_history'],
               'training_subjects': meta['training_subjects'], 'training_samples': meta['training_samples'],
               'device_memory': meta['device_memory'],
               'training_seconds': meta['training_seconds'], 'validation_seconds': meta['validation_seconds']}
        write_json(folder / 'budget.json', {'wall_seconds': elapsed, 'cap_seconds': CAP,
                                           'within_cap': elapsed <= CAP})
        if meta['optimizer_steps'] != (16 if pilot else 2148) or elapsed > CAP:
            raise RuntimeError('Schedule/budget failed; preserve evidence')
        return row
    with patch('numpy.load', side_effect=guarded_load):
        pilots = []
        for name, method, ablation in VARIANTS:
            pilots.append(run(name, method, ablation, pilot=True, experiment_hash=design['design_hash']))
            write_json(ROOT / 'pilot.partial.json', pilots)
        nval = sum(r['shape'][0] - 2 for r in manifest['subjects'] if r['split'] == 'val')
        projections = [{'variant': r['variant'], 'projected_seconds': 1.5 * (
            r['wall_seconds'] - r['training_seconds'] + 2 * nval * r['validation_seconds']
            + 2148 * r['training_seconds'] / r['updates'])} for r in pilots]
        protocol = {'design_hash': design['design_hash'], 'pilots': pilots,
                    'projections': projections, 'budget_seconds_per_run': CAP}
        protocol['protocol_hash'] = manifest_hash(protocol)
        write_json(ROOT / 'protocol.json', protocol)
        if any(r['projected_seconds'] > CAP for r in projections):
            raise RuntimeError('Pilot projection exceeds cap; no production training')
        rows = []
        for name, method, ablation in VARIANTS:
            rows.append(run(name, method, ablation, experiment_hash=protocol['protocol_hash']))
            write_json(ROOT / 'summary.partial.json', rows)
    write_json(ROOT / 'array_access_audit.json', {'load_count': len(loads),
               'allowed': sorted(allowed), 'loaded_paths': sorted(set(loads)),
               'forbidden_loads': 0, 'scope': 'All pilot and production np.load calls guarded'})
    write_json(ROOT / 'summary.json', {'protocol_hash': protocol['protocol_hash'], 'runs': rows,
               'test_scoring_performed': False, 'clean_reference_arrays_loaded': False})
    print('COMPLETE:', ROOT / 'summary.json')


if __name__ == '__main__':
    main()
