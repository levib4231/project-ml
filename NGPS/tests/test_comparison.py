from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.preparation import prepare
from ngps.comparison import comparison_pilot
from ngps.losses import NGPSHybridLoss
from ngps.training import train
from scripts.synthetic_pipeline import make_sources


class ComparisonTests(unittest.TestCase):
    def test_loss_ablation_arithmetic_and_default(self):
        center = torch.ones(1, 1, 3, 3, requires_grad=True)
        zero = torch.zeros_like(center)
        mask = torch.zeros_like(center)
        mask[:, :, 0] = 1
        args = (center, zero, zero, zero + 2, zero + 2, mask, mask, zero, zero)
        full = NGPSHybridLoss()(*args)
        no_rc = NGPSHybridLoss(0)(*args)
        no_dynamic = NGPSHybridLoss(include_dynamic=False)(*args)
        torch.testing.assert_close(full.total, full.n2n + full.dynamic + .5 * full.rc)
        torch.testing.assert_close(no_rc.total, full.n2n + full.dynamic)
        torch.testing.assert_close(no_dynamic.total, full.n2n + .5 * full.rc)
        self.assertEqual(no_dynamic.dynamic.item(), 0)
        no_dynamic.total.backward()
        self.assertTrue(torch.isfinite(center.grad).all())

    def setup_data(self, root):
        source = make_sources(root / 'sources')
        content = json.loads(source.read_text())
        next(r for r in content['subjects'] if r['split'] == 'train')['subject_id'] = 'subject04'
        source.write_text(json.dumps(content))
        manifest = prepare(source, root / 'prepared', 'already-unit')
        config = yaml.safe_load((ROOT / 'configs/synthetic_cpu.yaml').read_text())
        config['training']['batch_size'] = 1
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        return manifest, path

    def test_noisy_only_matched_pilot_and_no_overwrite(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest, config = self.setup_data(root)
            original_load = np.load
            def guarded(path, *args, **kwargs):
                # Reject clean arrays, source arrays, and all test observations.
                data = json.loads(Path(manifest).read_text())
                forbidden = {str((Path(manifest).parent / r['noisy']['0.06']['path']).resolve())
                             for r in data['subjects'] if r['split'] == 'test'}
                self.assertNotIn(str(Path(path).resolve()), forbidden)
                self.assertTrue(Path(path).name.startswith('noisy_'), str(path))
                return original_load(path, *args, **kwargs)
            with patch('numpy.load', side_effect=guarded):
                summary = comparison_pilot(manifest, config, root / 'pilot', 'cpu', 1, (1, 2))
            report = json.loads(summary.read_text())
            self.assertEqual(len(report['runs']), 10)
            self.assertFalse(report['test_scoring_performed'])
            self.assertEqual([r['optimizer_steps'] for r in report['runs']], [1] * 5 + [2] * 5)
            for r in report['runs']:
                self.assertIn('gradient_l2_norm', r['training_history'][-1])
                self.assertTrue(r['within_cap'])
            with self.assertRaises(FileExistsError):
                comparison_pilot(manifest, config, root / 'pilot', 'cpu', 1, (1,))

    def test_diagnostics_do_not_change_training(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest, config = self.setup_data(root)
            checkpoints = [train(manifest, config, root / str(flag), max_train_batches=1,
                                 max_val_batches=1, gradient_diagnostics=flag)
                           for flag in (False, True)]
            states = [torch.load(p, weights_only=False)['model'] for p in checkpoints]
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
