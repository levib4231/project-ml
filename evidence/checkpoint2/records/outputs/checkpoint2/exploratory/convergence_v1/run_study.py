"""Reproducible frozen noisy-only MPS convergence follow-up; run from repo root."""
from pathlib import Path
import copy
import json
import math
import sys
import time
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path('NGPS').resolve()))
from datasets.manifest import load_manifest, manifest_hash
from ngps.checkpoint2 import pilot, read_json
from ngps.device import synchronize
from ngps.provenance import provenance, source_record, write_json
from ngps.training import train

ROOT = Path(__file__).resolve().parent
MANIFEST = Path('outputs/checkpoint2/medical/mps_full_prepared/manifest.json').resolve()
BASE = Path('outputs/checkpoint2/medical/mps_full_run/training.yaml').resolve()
SCHEDULES = [(531, 1, 531), (1062, 2, 531), (2148, 2, 1074)]
METHODS = ['ngps', 'blindspot', 'adjacent']
CAP = 3600


def main():
    if (ROOT / 'design.json').exists():
        raise FileExistsError('Study already started; preserve its evidence')
    manifest = load_manifest(MANIFEST, prepared=True)
    config = yaml.safe_load(BASE.read_text())
    source = source_record()
    design = {'kind': 'exploratory_noisy_only_convergence', 'device': 'mps',
              'manifest_path': str(MANIFEST), 'manifest_hash': manifest_hash(manifest),
              'source_tree_sha256': source['tree_sha256'], 'hardware': provenance('mps'),
              'sigma': 0.06, 'methods': METHODS, 'base_config': config,
              'schedules': [{'updates': steps, 'epochs': epochs, 'batches_per_epoch': batches}
                            for steps, epochs, batches in SCHEDULES],
              'budget_seconds_per_run': CAP,
              'budget_reason': 'Independent MPS convergence follow-up with longer matched schedules and full noisy validation; pilot projection must pass before production. Not a transferred T4 budget.',
              'selection': 'Fixed final epoch; all schedules reported; no winner/model selection',
              'validation': 'All interior slices of subjects05/38; fixed masked noisy monitoring only',
              'training': 'All six agreed training subjects; full dataset shuffle; identical initialization, deterministic crops and sample order at shared epochs across objectives',
              'test_scoring': False, 'clean_reference_tuning': False,
              'limitations': 'Sigma0.06 only, seed42 only; noisy validation is not unbiased clean-risk or a cross-objective ranking. Schedules restart independently; not resumes. 531/1062 share a 531-batch epoch cap; 2148 uses all1074 batches/epoch.'}
    design['design_hash'] = manifest_hash(design)
    write_json(ROOT / 'design.json', design)
    for steps, epochs, batches in SCHEDULES:
        cfg = copy.deepcopy(config)
        cfg['training']['epochs'] = epochs
        (ROOT / f'updates_{steps}.yaml').write_text(yaml.safe_dump(cfg))
    # Reject every array load except train/validation noisy observations. This
    # wraps pilot and production, not just a declarative metadata assertion.
    allowed = {str((MANIFEST.parent / row['noisy']['0.06']['path']).resolve())
               for row in manifest['subjects'] if row['split'] in ('train', 'val')}
    real_load = np.load
    loads = []
    def guarded_load(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        if resolved not in allowed:
            raise RuntimeError('Forbidden array access: ' + resolved)
        loads.append(resolved)
        return real_load(path, *args, **kwargs)
    with patch('numpy.load', side_effect=guarded_load):
        pilot_path = pilot(MANIFEST, BASE, ROOT / 'pilot', device='mps')
        measurements = read_json(pilot_path)['measurements']
        step = max(r['training_seconds'] / r['optimizer_steps'] for r in measurements.values())
        overhead = max(r['wall_seconds'] - r['training_seconds'] for r in measurements.values())
        validation = max(r['validation_seconds'] for r in measurements.values())
        nval = sum(row['shape'][0] - 2 for row in manifest['subjects'] if row['split'] == 'val')
        projection = [{'updates': steps, 'projected_seconds': 1.5 * (overhead + epochs * nval * validation + steps * step)}
                      for steps, epochs, _ in SCHEDULES]
        if any(r['projected_seconds'] > CAP for r in projection):
            write_json(ROOT / 'projection_failed.json', projection)
            raise RuntimeError('Pilot projection exceeds cap; no production training')
        frozen = {'design_hash': design['design_hash'], 'pilot': read_json(pilot_path),
                  'projections': projection, 'budget_seconds_per_run': CAP}
        frozen['protocol_hash'] = manifest_hash(frozen)
        write_json(ROOT / 'protocol.json', frozen)
        rows = []
        for steps, epochs, batches in SCHEDULES:
            for method in METHODS:
                if source_record()['tree_sha256'] != source['tree_sha256']:
                    raise RuntimeError('Source changed after freeze')
                run = ROOT / 'runs' / f'{method}_updates{steps}'
                synchronize('mps')
                start = time.perf_counter()
                checkpoint = train(MANIFEST, ROOT / f'updates_{steps}.yaml', run,
                                   device='mps', sigma=0.06, method=method,
                                   max_train_batches=batches, experiment_hash=frozen['protocol_hash'],
                                   gradient_diagnostics=True)
                synchronize('mps')
                elapsed = time.perf_counter() - start
                meta = read_json(checkpoint.parent / 'metadata.json')
                row = {'method': method, 'updates': meta['optimizer_steps'], 'epochs': epochs,
                       'wall_seconds': elapsed, 'within_cap': elapsed <= CAP,
                       'training_history': meta['training_history'],
                       'validation_history': meta['validation_history'],
                       'training_subjects': meta['training_subjects'],
                       'device_memory': meta['device_memory']}
                rows.append(row)
                write_json(run / 'budget.json', {'wall_seconds': elapsed, 'cap_seconds': CAP,
                                                 'within_cap': elapsed <= CAP})
                write_json(ROOT / 'summary.partial.json', rows)
                if meta['optimizer_steps'] != steps or elapsed > CAP:
                    raise RuntimeError('Schedule/budget failed; preserve evidence')
    write_json(ROOT / 'array_access_audit.json', {'allowed': sorted(allowed),
               'load_count': len(loads), 'loaded_paths': sorted(set(loads)),
               'forbidden_loads': 0, 'scope': 'pilot and all production training'})
    write_json(ROOT / 'summary.json', {'protocol_hash': frozen['protocol_hash'], 'runs': rows,
               'test_scoring_performed': False, 'clean_reference_arrays_loaded': False})
    print('COMPLETE:', ROOT / 'summary.json')


if __name__ == '__main__':
    main()
