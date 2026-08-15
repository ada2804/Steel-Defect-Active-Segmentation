"""
Unit tests for RLE encoding and decoding utilities in src.rle_utils.
"""

import unittest
import numpy as np
from src.rle_utils import rle_to_mask, mask_to_rle


class TestRLEUtils(unittest.TestCase):
    """Test suite for RLE conversion functions."""

    def test_empty_and_null_inputs(self):
        """Test that empty, None, and NaN inputs produce all-zero masks."""
        h, w = 256, 1600
        mask_none = rle_to_mask(None, width=w, height=h)
        self.assertEqual(mask_none.shape, (h, w))
        self.assertEqual(np.sum(mask_none), 0)

        mask_nan = rle_to_mask(float("nan"), width=w, height=h)
        self.assertEqual(np.sum(mask_nan), 0)

        mask_empty = rle_to_mask("", width=w, height=h)
        self.assertEqual(np.sum(mask_empty), 0)

        rle_empty = mask_to_rle(np.zeros((h, w), dtype=np.uint8))
        self.assertEqual(rle_empty, "")

    def test_simple_rle_roundtrip(self):
        """Test roundtrip encoding and decoding on known shapes."""
        h, w = 256, 1600
        # Create a synthetic mask with specific rectangular defect regions
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:30, 50:70] = 1
        mask[100:150, 400:450] = 1

        rle = mask_to_rle(mask)
        self.assertIsInstance(rle, str)
        self.assertGreater(len(rle), 0)

        reconstructed_mask = rle_to_mask(rle, width=w, height=h)
        self.assertTrue(np.array_equal(mask, reconstructed_mask))

    def test_known_pattern(self):
        """Test specific known Fortran-order indices."""
        # 4x4 matrix
        # Column 0: pixels 1, 2, 3, 4
        # Pixel (row=0, col=0) is index 1
        # Pixel (row=2, col=0) is index 3
        # Pixel (row=1, col=1) is index 6 (4 + 2)
        h, w = 4, 4
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0, 0] = 1  # 1
        mask[2, 0] = 1  # 3
        mask[1, 1] = 1  # 6

        rle = mask_to_rle(mask)
        expected_rle = "1 1 3 1 6 1"
        self.assertEqual(rle, expected_rle)

        reconstructed = rle_to_mask(rle, width=w, height=h)
        self.assertTrue(np.array_equal(mask, reconstructed))


if __name__ == "__main__":
    unittest.main()
