"""
Unit tests for SeverstalDataset in src.dataset.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import SeverstalDataset
from src.rle_utils import mask_to_rle


class TestSeverstalDataset(unittest.TestCase):
    """Test suite for SeverstalDataset loading and transformation pipeline."""

    def setUp(self):
        """Create temporary test directory and synthetic images and annotations."""
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.temp_dir, "train_images")
        os.makedirs(self.img_dir, exist_ok=True)

        # Create 2 synthetic test images: one normal, one with defect
        self.img1_name = "test_normal.jpg"
        self.img2_name = "test_defect.jpg"

        img1 = Image.fromarray(np.full((256, 1600, 3), 120, dtype=np.uint8))
        img1.save(os.path.join(self.img_dir, self.img1_name))

        img2 = Image.fromarray(np.full((256, 1600, 3), 150, dtype=np.uint8))
        img2.save(os.path.join(self.img_dir, self.img2_name))

        # Create a defect mask for img2, class 3
        defect_mask = np.zeros((256, 1600), dtype=np.uint8)
        defect_mask[50:100, 200:300] = 1
        rle_class3 = mask_to_rle(defect_mask)

        # Create CSV records (legacy schema ImageId_ClassId, EncodedPixels)
        records = [
            {"ImageId_ClassId": f"{self.img1_name}_1", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img1_name}_2", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img1_name}_3", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img1_name}_4", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img2_name}_1", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img2_name}_2", "EncodedPixels": ""},
            {"ImageId_ClassId": f"{self.img2_name}_3", "EncodedPixels": rle_class3},
            {"ImageId_ClassId": f"{self.img2_name}_4", "EncodedPixels": ""},
        ]
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        pd.DataFrame(records).to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_loading_and_shapes(self):
        """Verify tensor shapes, types, and defect detection."""
        dataset = SeverstalDataset(img_dir=self.img_dir, csv_path=self.csv_path)
        self.assertEqual(len(dataset), 2)

        # Check normal sample
        sample_normal = dataset[0] if dataset.image_ids[0] == self.img1_name else dataset[1]
        self.assertEqual(sample_normal["image"].shape, torch.Size([3, 224, 224]))
        self.assertEqual(sample_normal["mask_16x16"].shape, torch.Size([4, 16, 16]))
        self.assertEqual(sample_normal["full_mask"].shape, torch.Size([4, 256, 1600]))
        self.assertFalse(sample_normal["has_defect"])
        self.assertEqual(torch.sum(sample_normal["mask_16x16"]), 0)

        # Check defective sample
        sample_defect = dataset[1] if dataset.image_ids[1] == self.img2_name else dataset[0]
        self.assertTrue(sample_defect["has_defect"])
        # Class 3 mask should have positive patches (index 2)
        self.assertGreater(torch.sum(sample_defect["mask_16x16"][2]), 0)
        # Other classes should be 0
        self.assertEqual(torch.sum(sample_defect["mask_16x16"][0]), 0)
        self.assertEqual(torch.sum(sample_defect["mask_16x16"][1]), 0)
        self.assertEqual(torch.sum(sample_defect["mask_16x16"][3]), 0)


if __name__ == "__main__":
    unittest.main()
