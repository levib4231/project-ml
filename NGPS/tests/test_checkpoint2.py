from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity

from datasets.preparation import prepare
from datasets.manifest import load_manifest, manifest_hash
from ngps.checkpoint2 import (benchmark, export_diagnostics, freeze, load_protocol,
                             preflight, prepare_verified, train_matrix, validate_budget)
from ngps.config import NGPSConfig
from ngps.inference import timed_volume_file
from ngps.metrics import hfen, psnr, ssim
from ngps.sampler import NeighborGuidedPatchSampler
from ngps.provenance import source_record
from scripts.synthetic_pipeline import make_sources

torch.set_num_threads(1)


class DiagnosticTests(unittest.TestCase):
    def test_coordinates_retrieve_original_raw_values_at_edges(self):
        cfg = NGPSConfig(patch_size=3, search_window=3, topk=2)
        sampler = NeighborGuidedPatchSampler(cfg)
        guide = torch.rand(2, 1, 5, 7)
        raw = torch.arange(70).reshape(2, 1, 5, 7).float() - 40
        original = raw.clone()
        detail = sampler.match_details(guide, guide.roll(1, -1), raw)
        for b in range(2):
            torch.testing.assert_close(detail['raw_values'][b], raw[b, 0][detail['y'][b], detail['x'][b]])
        torch.testing.assert_close(detail['target'], detail['raw_values'].mean(1, keepdim=True))
        torch.testing.assert_close(raw, original)
        self.assertFalse(detail['target'].requires_grad)

    def test_metrics_against_independent_calculations(self):
        ref = np.random.default_rng(4).random((12, 14), dtype=np.float32)
        pred = ref + np.float32(0.05)
        self.assertAlmostEqual(psnr(ref, pred), 10 * np.log10(1 / np.mean((ref-pred).astype(float)**2)), places=5)
        self.assertAlmostEqual(ssim(ref, pred), structural_similarity(ref, pred, data_range=1, win_size=7))
        expected = np.linalg.norm(gaussian_laplace(ref, 1.5)-gaussian_laplace(pred,1.5)) / (np.linalg.norm(gaussian_laplace(ref,1.5))+1e-8)
        self.assertAlmostEqual(hfen(ref,pred), expected)
    def test_ssim_smallest_supported_shape(self):
        image = np.ones((3, 3), dtype=np.float32)
        self.assertAlmostEqual(ssim(image, image), 1.0)

        too_small = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "spatial dimensions >= 3"):
            ssim(too_small, too_small)

    def test_load_to_save_timer_order_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp/'noisy.npy'
            np.save(path, np.ones((3, 8, 8), dtype=np.float32))
            times = iter([0., 1., 3., 5.])
            with patch('ngps.inference.time.perf_counter', side_effect=lambda: next(times)):
                result, timing = timed_volume_file(lambda a: a*2, path, tmp/'prediction.npy')
            self.assertEqual(timing, {'inference_seconds':2., 'volume_seconds':5.})
            np.testing.assert_array_equal(np.load(tmp/'prediction.npy'), result)
            with self.assertRaises(FileExistsError):
                timed_volume_file(lambda a:a, path, tmp/'prediction.npy')


class ExperimentTests(unittest.TestCase):
    def setup_experiment(self, root):
        source = make_sources(root/'sources')
        manifest = prepare(source, root/'prepared', 'already-unit')
        protocol = freeze(manifest, ROOT/'configs/synthetic_cpu.yaml', root/'experiment', synthetic=True)
        return manifest, protocol

    def test_freeze_rejects_changed_configuration_and_medical_shortcuts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest, protocol=self.setup_experiment(root)
            with self.assertRaises(ValueError):
                freeze(manifest, ROOT/'configs/synthetic_cpu.yaml', root/'invalid', synthetic=False)
            content=json.loads(protocol.read_text())
            content['tolerance_db']=1.0
            protocol.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'modified'):
                load_protocol(protocol)

    def test_receipt_must_bind_source_and_every_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=make_sources(root/'sources')
            receipt=root/'receipt.json'
            receipt.write_text(json.dumps({'status':'verified_fixed_reference',
                'source_manifest_hash':manifest_hash(load_manifest(source)),
                'normalization':{'policy':'already-unit'},'subjects':[]}))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                prepare_verified(source, receipt, root/'prepared')

    def test_missing_matrix_rejected_before_clean_or_test_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, protocol=self.setup_experiment(Path(tmp))
            with patch('numpy.load', side_effect=AssertionError('Array opened before complete preflight')):
                with self.assertRaisesRegex(ValueError,'Missing selected'):
                    preflight(protocol)

    def test_nine_runs_and_complete_benchmark_with_leakage_guards(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            manifest_path, protocol_path=self.setup_experiment(root)
            original_load=np.load
            def training_only(path,*args,**kwargs):
                name=Path(path).name
                self.assertTrue(name.startswith('noisy_'), str(path))
                # Check exact test paths rather than subject numbering conventions.
                test_paths={str((manifest_path.parent/r['noisy'][f'{s:.2f}']['path']).resolve())
                            for r in load_manifest(manifest_path)['subjects'] if r['split']=='test'
                            for s in (0.03,0.06,0.10)}
                self.assertNotIn(str(Path(path).resolve()),test_paths)
                return original_load(path,*args,**kwargs)
            with patch('numpy.load', side_effect=training_only):
                train_matrix(protocol_path,'cpu')
            checkpoints=list((protocol_path.parent/'runs').glob('*/model_final.pth'))
            self.assertEqual(len(checkpoints),9)
            # A completed rerun skips valid checkpoints.
            with patch('ngps.checkpoint2.train', side_effect=AssertionError('Unexpected retraining')):
                train_matrix(protocol_path,'cpu')
            protocol,manifest,_,_=preflight(protocol_path)
            # Rebind one selected checkpoint to the wrong objective; the matrix is rejected.
            first=checkpoints[0]
            saved=first.read_bytes()
            state=torch.load(first,map_location='cpu',weights_only=True)
            state['metadata']['method']='invalid'
            torch.save(state,first)
            with patch('numpy.load',side_effect=AssertionError('Reference opened for invalid matrix')):
                with self.assertRaises(ValueError): preflight(protocol_path)
            first.write_bytes(saved)
            # Diagnostics accept training subjects only and never open references.
            subject=next(r['subject_id'] for r in manifest['subjects'] if r['split']=='train')
            with patch('numpy.load', side_effect=training_only):
                export_diagnostics(manifest_path,protocol_path.parent/'training.yaml',root/'diagnostic.npz',subject)
            result=benchmark(protocol_path,'cpu')
            report=json.loads((result/'benchmark.json').read_text())
            self.assertEqual(len(report['subjects']),30)
            self.assertEqual(len(report['success_by_sigma']),3)
            self.assertEqual(len(list((result/'predictions').glob('*/*.npy'))),30)
            self.assertEqual(len(list((result/'figures').glob('*.png'))),6)
            self.assertTrue((result/'COMPLETE.json').exists())
            self.assertTrue(all(row['slices']==4 for row in report['subjects']))
            self.assertTrue(all(row['volume_seconds']>=row['inference_seconds']>0 for row in report['subjects']))
            for target in report['success_by_sigma']:
                self.assertEqual(target['target_met'],target['ngps_psnr']>=target['comparator_psnr'])
            with self.assertRaisesRegex(ValueError,'testing already started'):
                train_matrix(protocol_path,'cpu')

    def test_budget_failure_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.json'
            path.write_text(json.dumps({'wall_seconds':700,'cap_seconds':600,'within_cap':True}))
            with self.assertRaises(ValueError):validate_budget(path,{'budget_seconds':600})

    def test_pilot_projects_full_validation_and_freezes_equal_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root/'sources')
            manifest = prepare(source, root/'prepared', 'already-unit')
            config = yaml.safe_load((ROOT/'configs/synthetic_cpu.yaml').read_text())
            config['training']['batch_size'] = 1
            config_path = root/'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            record = {'manifest_hash': manifest_hash(load_manifest(manifest)),
                      'source_tree_sha256': source_record()['tree_sha256'],
                      'config': config, 'device': 'cpu',
                      'measurements': {method: {'training_seconds': 1.6, 'optimizer_steps': 16,
                                               'wall_seconds': 2.0, 'validation_seconds': 0.01}
                                       for method in ('ngps','blindspot','adjacent')}}
            pilot_path = root/'pilot.json'
            pilot_path.write_text(json.dumps(record))
            protocol = freeze(manifest, config_path, root/'experiment', pilot_path,
                              budget_seconds=10, synthetic=True)
            frozen, _, _ = load_protocol(protocol)
            self.assertEqual(frozen['max_train_batches'], 12)
            self.assertGreater(frozen['pilot_projection']['projected_seconds'], 1.8)
            with self.assertRaisesRegex(ValueError, 'exhausts the budget'):
                freeze(manifest, config_path, root/'too_small', pilot_path,
                       budget_seconds=0.1, synthetic=True)
            with self.assertRaisesRegex(ValueError, 'requires a reason'):
                freeze(manifest, config_path, root/'too_large', pilot_path,
                       budget_seconds=601, synthetic=True)


if __name__=='__main__': unittest.main()
