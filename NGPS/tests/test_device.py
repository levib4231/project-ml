from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ngps.device import device_name, is_accelerator, memory_stats, preferred_device, require_device, synchronize


class DeviceTests(unittest.TestCase):
    def test_cpu_device_contract(self):
        device = require_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertFalse(is_accelerator(device))
        synchronize(device)
        stats = memory_stats(device)
        self.assertEqual(stats["backend"], "cpu")
        self.assertTrue(device_name(device))

    def test_preferred_device_is_supported(self):
        self.assertIn(preferred_device(), {"cpu", "cuda", "mps"})

    @unittest.skipUnless(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available(),
        "Apple MPS is unavailable",
    )
    def test_mps_device_contract_and_memory_telemetry(self):
        device = require_device("mps")
        self.assertTrue(is_accelerator(device))
        tensor = torch.ones(4, device=device)
        synchronize(device)
        stats = memory_stats(device)
        self.assertEqual(stats["backend"], "mps")
        self.assertIsNone(stats["peak_memory_bytes"])
        self.assertGreaterEqual(stats["current_memory_bytes"], 0)
        self.assertGreaterEqual(stats["driver_memory_bytes"], stats["current_memory_bytes"])
        del tensor


if __name__ == "__main__":
    unittest.main()
