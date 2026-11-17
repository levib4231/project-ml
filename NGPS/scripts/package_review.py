"""Package small local review evidence, never volumes, predictions or weights.

Run from the repository root. Nothing is uploaded. Original outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPLORATORY = ('comparison_pilot_v1', 'convergence_v1', 'loss_ablation_v1',
               'rician_v1', 'noise_isolation_v1')
TOP_FILES = ('REPORT.md', 'run_study.py', 'finalize_report.py', 'design.json',
             'protocol.json', 'summary.json', 'training.yaml', 'pilot.yaml',
             'array_access_audit.json', 'training_access_audit.json', 'noise_tests.json',
             'reporting_recovery.json', 'retrieval_diagnostics.json', 'test_started.json')
RESULT_FILES = ('benchmark.json', 'subjects.csv', 'aggregates.csv', 'success.csv',
                'worst_slices.csv', 'COMPLETE.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_files():
    files = set()
    medical = ROOT / 'outputs/checkpoint2/medical'
    for experiment in (medical, medical / 'mps_full_run', ROOT / 'outputs/checkpoint2/exploratory/rician_v1/experiment'):
        for name in ('protocol.json', 'training.yaml', 'test_started.json'):
            files.add(experiment / name)
        files.update(experiment / 'results' / name for name in RESULT_FILES)
        files.update(experiment.glob('runs/*/metadata.json'))
        files.update(experiment.glob('runs/*/budget.json'))
    for name in ('pilot.json', 'WORKLOAD_DECISION.json', 'checkpoint2_budgeted.yaml', 'brainweb_verification.json'):
        files.add(medical / 'evidence' / name)
    for pilot in ('mps_full_pilot',):
        files.update((medical / pilot).glob('*.json'))
        files.update((medical / pilot).glob('*.yaml'))
    files.add(ROOT / 'outputs/checkpoint2/local-verification/brainweb_verification.json')
    for prep in (medical / 'evidence/prepared', medical / 'mps_full_prepared',
                 ROOT / 'outputs/checkpoint2/exploratory/rician_v1/prepared'):
        files.add(prep / 'manifest.json')
    files.add(ROOT / 'outputs/checkpoint2/exploratory/experiment_index.json')
    for study in EXPLORATORY:
        folder = ROOT / 'outputs/checkpoint2/exploratory' / study
        files.update(folder / name for name in TOP_FILES)
        if study == 'noise_isolation_v1':
            # Preserve the exact inputs to the reporting-only recovery.
            files.add(folder / 'scores.partial.json')
        files.update(folder.glob('*.yaml'))
        files.update(folder.glob('runs/**/metadata.json'))
        files.update(folder.glob('runs/**/budget.json'))
        files.update(folder.glob('runs/**/pilot_budget.json'))
        files.update(folder.glob('pilot/**/metadata.json'))
        files.update(folder.glob('pilot/**/budget.json'))
        files.update(folder.glob('pilot/*.json'))
        files.update(folder.glob('pilot/*.yaml'))
    return sorted(path for path in files if path.is_file())


def verify_package(package):
    package = Path(package)
    inventory = json.loads((package / 'inventory.json').read_text())
    entries = inventory['files']
    allowed = {'.json', '.csv', '.yaml', '.yml', '.md', '.py', '.zip'}
    for name, record in entries.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.suffix not in allowed:
            raise ValueError('Unsafe or prohibited package path: ' + name)
        path = package / rel
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Package hash mismatch: ' + name)
        if rel.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt source archive')
                for member in archive.namelist():
                    part = Path(member)
                    if part.is_absolute() or '..' in part.parts or part.suffix in {'.npy','.npz','.mnc','.gz','.pth','.pt','.ckpt'}:
                        raise ValueError('Prohibited source archive member: ' + member)
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()} - {'inventory.json'}
    if actual != set(entries):
        raise ValueError('Uninventoried or missing package files')
    return {'files': len(entries), 'bytes': sum(r['bytes'] for r in entries.values()), 'status': 'verified'}


def package_review(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = selected_files()
    origins = {}
    snapshots = {}
    for path in sources:
        rel = Path('records') / path.relative_to(ROOT)
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        origins[str(rel)] = str(path.relative_to(ROOT))
        if path.name == 'metadata.json':
            meta = json.loads(path.read_text())
            tree = meta['source']['tree_sha256']
            snapshot = path.parent / 'source_snapshot.zip'
            if snapshot.is_file():
                snapshots.setdefault(tree, snapshot)
    for tree, path in snapshots.items():
        rel = Path('historical_sources') / (tree + '.zip')
        dest = output / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        origins[str(rel)] = str(path.relative_to(ROOT))
    guide = ROOT / 'NGPS/docs/SUBMISSION_EVIDENCE.md'
    shutil.copyfile(guide, output / 'README.md')
    origins['README.md'] = str(guide.relative_to(ROOT))
    inventory = {'schema_version': 1, 'scope': 'Local small review evidence; no medical arrays, predictions, weights or figures',
                 'files': {name: {'sha256': digest(output / name), 'bytes': (output / name).stat().st_size,
                                  'origin': origin} for name, origin in sorted(origins.items())}}
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return verify_package(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='evidence/checkpoint2')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_package(args.output) if args.verify else package_review(args.output), indent=2))
