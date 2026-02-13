"""
Tests for FP8 MPS Metal support module.
"""

import unittest


class TestFP8MPS(unittest.TestCase):
    """Test FP8 MPS module import and basic functionality."""

    def test_module_import(self):
        """Test that the fp8_mps module can be imported."""
        try:
            import comfy.fp8_mps
            self.assertTrue(hasattr(comfy.fp8_mps, 'install'))
            self.assertTrue(hasattr(comfy.fp8_mps, 'uninstall'))
            self.assertTrue(hasattr(comfy.fp8_mps, 'is_installed'))
        except ImportError as e:
            self.fail(f"Failed to import comfy.fp8_mps: {e}")

    def test_patch_functions_exist(self):
        """Test that patch functions are callable."""
        import comfy.fp8_mps
        self.assertTrue(callable(comfy.fp8_mps.install))
        self.assertTrue(callable(comfy.fp8_mps.uninstall))
        self.assertTrue(callable(comfy.fp8_mps.is_installed))


if __name__ == '__main__':
    unittest.main()
