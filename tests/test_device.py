import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from corridorkey.device import resolve_accelerator


@unittest.skipIf(torch is None, "PyTorch is required for device tests")
class DeviceResolutionTests(unittest.TestCase):
    @patch("corridorkey.device.torch.cuda.is_available", return_value=True)
    @patch("corridorkey.device.torch.version.cuda", "12.8")
    @patch("corridorkey.device.torch.version.hip", None)
    def test_auto_selects_nvidia(self, _hip, _cuda, _available):
        accelerator = resolve_accelerator()
        self.assertEqual(accelerator.backend, "nvidia")
        self.assertEqual(accelerator.device, torch.device("cuda"))

    @patch("corridorkey.device.torch.cuda.is_available", return_value=True)
    @patch("corridorkey.device.torch.version.cuda", None)
    @patch("corridorkey.device.torch.version.hip", "6.3")
    def test_auto_selects_amd(self, _hip, _cuda, _available):
        self.assertEqual(resolve_accelerator().backend, "amd")

    @patch("corridorkey.device.torch.cuda.is_available", return_value=False)
    def test_auto_requires_accelerator(self, _available):
        with self.assertRaisesRegex(RuntimeError, "no supported GPU backend"):
            resolve_accelerator()

    @patch("corridorkey.device.torch.cuda.is_available", return_value=True)
    @patch("corridorkey.device.torch.version.cuda", "12.8")
    @patch("corridorkey.device.torch.version.hip", None)
    def test_explicit_amd_rejects_cuda_build(self, _hip, _cuda, _available):
        with self.assertRaisesRegex(RuntimeError, "--gpu amd"):
            resolve_accelerator("amd")


if __name__ == "__main__":
    unittest.main()
