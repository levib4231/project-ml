"""Reporting-only recovery of subject-key mismatch; no training or rescoring."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path('NGPS').resolve()))
from datasets.manifest import manifest_hash
from datasets.preparation import file_hash
from ngps.checkpoint2 import read_json
from ngps.metrics import aggregate_subjects
from ngps.provenance import source_record, write_json

root = Path(__file__).resolve().parent
design = read_json(root / 'design.json')
claimed = design.pop('design_hash')
assert claimed == manifest_hash(design)
assert file_hash(root / 'run_study.py') == design['orchestrator_sha256']
assert source_record()['tree_sha256'] == design['source_tree_sha256']
protocol = read_json(root / 'protocol.json')
claimed_protocol = protocol.pop('protocol_hash')
assert claimed_protocol == manifest_hash(protocol)
runs = read_json(root / 'training.partial.json')
scores = read_json(root / 'scores.partial.json')
assert len(runs) == 8 and len(scores) == 20
assert len({(r['noise'],r['variant'],r['subject']) for r in scores}) == 20
for r in runs:
    assert r['updates'] == 531 and r['within_cap']
    assert file_hash(r['checkpoint']) == r['checkpoint_sha256']
aggregates = []
normalized = [{**row, 'subject_id': row['subject']} for row in scores]
for noise in ('gaussian', 'rician'):
    for variant in ('full_ngps','no_rc','no_dynamic','adjacent','noisy'):
        group = [r for r in normalized if r['noise'] == noise and r['variant'] == variant]
        assert {r['subject_id'] for r in group} == {'subject20','subject44'}
        aggregates.append({'noise': noise, 'variant': variant, 'metrics': aggregate_subjects(group),
                           'error_strata': {region: {key: float(np.mean([r['error_strata'][region][key] for r in group]))
                           for key in ('mean_signed_error','mse','spatial_centered_error_variance')}
                           for region in group[0]['error_strata']}})
write_json(root / 'reporting_recovery.json', {'failure': 'KeyError subject_id during summary aggregation after all frozen scoring',
           'fix': 'Reporting-only subject-to-subject_id alias; unchanged original scores.partial.json and all predictions/checkpoints',
           'rescored': False, 'retrained': False, 'original_scores_sha256': file_hash(root / 'scores.partial.json'),
           'finalizer_sha256': file_hash(__file__)})
write_json(root / 'summary.json', {'protocol_hash': claimed_protocol, 'runs': runs,
           'subjects': normalized, 'aggregates': aggregates, 'limitations': design['limitations'],
           'prediction_sha256': {str(p.relative_to(root)): file_hash(p) for p in sorted((root/'predictions').glob('*/*/*.npy'))}})
print('Summary complete; reporting-only recovery')
