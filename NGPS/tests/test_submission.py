import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_review import digest, verify_package


class SubmissionTests(unittest.TestCase):
    def fixture(self, root, name='report.json'):
        path = root / name
        path.write_text('{}\n')
        (root / 'inventory.json').write_text(json.dumps({'files': {name: {
            'sha256': digest(path), 'bytes': path.stat().st_size}}}))
        return path

    def test_inventory_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.fixture(root)
            self.assertEqual(verify_package(root)['files'], 1)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                verify_package(root)

    def test_uninventoried_files_and_prohibited_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'extra.npy').write_bytes(b'not allowed')
            with self.assertRaises(ValueError):
                verify_package(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 'volume.npy')
            with self.assertRaises(ValueError):
                verify_package(root)
