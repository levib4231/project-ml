"""Read-only audit of packaged historical results and optional local large artifacts."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import manifest_hash
from datasets.preparation import file_hash
from ngps.metrics import aggregate_subjects
from ngps.provenance import source_record, write_json
from scripts.package_review import verify_package

ROOT = Path(__file__).resolve().parents[2]


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Numerical mismatch: {a} != {b}')


def verify_review(package, local=False):
    package = Path(package).resolve()
    result = {'package': verify_package(package), 'benchmarks': [], 'source_archives': [],
              'historical_source_warning': 'Historical protocols must be run with their archived source, not relabeled with the current source hash.',
              'current_source_tree_sha256': source_record()['tree_sha256']}
    records = package / 'records'
    for path in sorted((package / 'historical_sources').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(archive.namelist())}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if tree != path.stem:
            raise ValueError('Historical source tree mismatch: ' + str(path))
        result['source_archives'].append(tree)
    for report_path in sorted(records.glob('outputs/checkpoint2/**/results/benchmark.json')):
        report = json.loads(report_path.read_text())
        experiment = report_path.parent.parent
        protocol = json.loads((experiment / 'protocol.json').read_text())
        claimed = protocol.pop('protocol_hash')
        if manifest_hash(protocol) != claimed or report['protocol_hash'] != claimed:
            raise ValueError('Protocol mismatch')
        if file_hash(experiment / 'training.yaml') != protocol['config_sha256']:
            raise ValueError('Frozen config mismatch')
        completion = json.loads((report_path.parent / 'COMPLETE.json').read_text())
        if completion['benchmark_sha256'] != file_hash(report_path):
            raise ValueError('Completion hash mismatch')
        with (report_path.parent / 'subjects.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(report['subjects']) != 30 or len(csv_rows) != 30:
            raise ValueError('Expected30 subject rows')
        for row, csv_row in zip(report['subjects'], csv_rows):
            if (row['subject_id'], row['method'], row['sigma']) != (csv_row['subject_id'], csv_row['method'], float(csv_row['sigma'])):
                raise ValueError('Subject table mismatch')
            for key in ('PSNR','SSIM','HFEN','inference_seconds','volume_seconds'):
                close(row[key], csv_row[key])
        for sigma in protocol['sigmas']:
            for method in protocol['methods'] + protocol['controls']:
                group = [r for r in report['subjects'] if r['sigma'] == sigma and r['method'] == method]
                computed = aggregate_subjects(group)
                for metric, stats in computed.items():
                    saved = next(r for r in report['aggregates'] if r['sigma'] == sigma and r['method'] == method and r['metric'] == metric)
                    for key in ('mean','std_population','n_subjects'):
                        close(stats[key], saved[key])
            means = {r['method']: r['mean'] for r in report['aggregates'] if r['sigma'] == sigma and r['metric'] == 'PSNR'}
            saved = next(r for r in report['success_by_sigma'] if r['sigma'] == sigma)
            strongest = max(('blindspot','adjacent'), key=means.get)
            close(saved['gap_db'], means['ngps'] - means[strongest])
            assert saved['strongest_comparator'] == strongest and saved['target_met'] == (means['ngps'] >= means[strongest])
        assert report['all_targets_met'] == all(r['target_met'] for r in report['success_by_sigma'])
        with (report_path.parent / 'success.csv').open() as stream:
            for a, b in zip(csv.DictReader(stream), report['success_by_sigma']):
                close(a['gap_db'], b['gap_db'])
                assert a['target_met'] == str(b['target_met'])
        with (report_path.parent / 'aggregates.csv').open() as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(report['aggregates'])
        for a, b in zip(saved, report['aggregates']):
            assert a['method'] == b['method'] and a['metric'] == b['metric']
            close(a['mean'], b['mean']); close(a['std_population'], b['std_population'])
        for run in sorted((experiment / 'runs').glob('*/metadata.json')):
            metadata = json.loads(run.read_text())
            workload = min(protocol['max_train_batches'], math.ceil(metadata['training_samples'] / metadata['training_config']['batch_size'])) * metadata['training_config']['epochs']
            assert metadata['optimizer_steps'] == workload
            assert metadata['experiment_hash'] == claimed
            assert metadata['source']['tree_sha256'] == protocol['source_tree_sha256']
            budget = json.loads((run.parent / 'budget.json').read_text())
            assert budget['within_cap'] and budget['wall_seconds'] <= protocol['budget_seconds']
        result['benchmarks'].append({'path': str(report_path.relative_to(records)), 'protocol_hash': claimed,
                                     'rows': 30, 'all_targets_met': report['all_targets_met']})
    result['exploratory_studies'] = []
    exploratory = records / 'outputs/checkpoint2/exploratory'
    for name in ('comparison_pilot_v1', 'convergence_v1', 'loss_ablation_v1', 'noise_isolation_v1'):
        folder = exploratory / name
        summary = json.loads((folder / 'summary.json').read_text())
        for row in summary['runs']:
            if name == 'comparison_pilot_v1':
                run = folder / 'runs' / row['run']
            elif name == 'convergence_v1':
                run = folder / 'runs' / f"{row['method']}_updates{row['updates']}"
            elif name == 'loss_ablation_v1':
                run = folder / 'runs' / row['variant']
            else:
                run = folder / 'runs' / row['noise'] / row['variant']
            metadata = json.loads((run / 'metadata.json').read_text())
            assert metadata['training_history'] == row['training_history']
            assert metadata['validation_history'] == row['validation_history']
            assert metadata['optimizer_steps'] == row.get('updates', row.get('optimizer_steps'))
            assert row['within_cap']
        protocol = json.loads((folder / 'protocol.json').read_text())
        claimed = protocol.pop('protocol_hash')
        assert manifest_hash(protocol) == claimed and summary['protocol_hash'] == claimed
        design_path = folder / 'design.json'
        if design_path.exists():
            design = json.loads(design_path.read_text())
            claimed_design = design.pop('design_hash')
            assert manifest_hash(design) == claimed_design and protocol['design_hash'] == claimed_design
            if 'orchestrator_sha256' in design:
                assert file_hash(folder / 'run_study.py') == design['orchestrator_sha256']
        if name == 'noise_isolation_v1':
            for aggregate in summary['aggregates']:
                group = [r for r in summary['subjects'] if r['noise'] == aggregate['noise'] and r['variant'] == aggregate['variant']]
                assert aggregate_subjects(group) == aggregate['metrics']
            recovery = json.loads((folder / 'reporting_recovery.json').read_text())
            assert file_hash(folder / 'scores.partial.json') == recovery['original_scores_sha256']
            assert file_hash(folder / 'finalize_report.py') == recovery['finalizer_sha256']
        result['exploratory_studies'].append({'study': name, 'runs_verified': len(summary['runs'])})
    # Manifests are evidence records only; relocated paths intentionally retain
    # original meanings. Never pass a relocated manifest to training/evaluation.
    if local:
        checks = {}
        for relative in ('outputs/checkpoint2/medical/mps_full_prepared/manifest.json',
                         'outputs/checkpoint2/exploratory/rician_v1/prepared/manifest.json'):
            path = ROOT / relative
            manifest = json.loads(path.read_text())
            count = 0
            for row in manifest['subjects']:
                for name, claimed in [(row['source'], row['source_sha256']), (row['clean'], row['clean_sha256'])] + [(e['path'], e['sha256']) for e in row['noisy'].values()]:
                    if file_hash((path.parent / name).resolve()) != claimed:
                        raise ValueError('Local data hash mismatch')
                    count += 1
            checks[relative] = {'manifest_hash': manifest_hash(manifest), 'files_verified': count}
        result['local_data'] = checks
        for relative in ('outputs/checkpoint2/medical', 'outputs/checkpoint2/medical/mps_full_run',
                         'outputs/checkpoint2/exploratory/rician_v1/experiment'):
            path = ROOT / relative
            report = json.loads((path / 'results/benchmark.json').read_text())
            for name, claimed in report['checkpoint_sha256'].items():
                assert file_hash(path / 'runs' / name / 'model_final.pth') == claimed
            for row in report['subjects']:
                prediction = path / 'results/predictions' / f"{row['method']}_s{row['sigma']:.2f}" / (row['subject_id'] + '.npy')
                assert file_hash(prediction) == row['prediction_sha256']
        result['large_artifacts'] = {'checkpoints_verified': 27, 'predictions_verified': 90}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='evidence/checkpoint2')
    parser.add_argument('--local', action='store_true', help='Also hash local volumes, weights and predictions; no array loading')
    parser.add_argument('--output', required=True, help='New report outside NGPS')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = verify_review(args.package, args.local)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
