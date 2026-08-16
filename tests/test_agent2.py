"""
Unit tests for AGENT2.md pipeline: Data Engine mining and Submission formatting.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.dataset import SeverstalUNetDataset
from src.rle_utils import rle_to_mask, mask_to_rle
from src.model import ProgressiveUNetDecoder


class TestAgent2Pipeline(unittest.TestCase):
    """Test suite for AGENT2 pipeline components."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.temp_dir, "test_imgs")
        os.makedirs(self.img_dir, exist_ok=True)

        # Create 2 synthetic test images
        img1 = Image.fromarray(np.full((256, 1600, 3), 100, dtype=np.uint8))
        img1.save(os.path.join(self.img_dir, "001.jpg"))

        img2 = Image.fromarray(np.full((256, 1600, 3), 180, dtype=np.uint8))
        img2.save(os.path.join(self.img_dir, "002.jpg"))

        # Create synthetic CSV
        defect_mask = np.zeros((256, 1600), dtype=np.uint8)
        defect_mask[20:80, 50:150] = 1
        rle = mask_to_rle(defect_mask)

        records = [
            {"ImageId_ClassId": "001.jpg_1", "EncodedPixels": ""},
            {"ImageId_ClassId": "001.jpg_2", "EncodedPixels": ""},
            {"ImageId_ClassId": "001.jpg_3", "EncodedPixels": ""},
            {"ImageId_ClassId": "001.jpg_4", "EncodedPixels": ""},
            {"ImageId_ClassId": "002.jpg_1", "EncodedPixels": rle},
            {"ImageId_ClassId": "002.jpg_2", "EncodedPixels": ""},
            {"ImageId_ClassId": "002.jpg_3", "EncodedPixels": ""},
            {"ImageId_ClassId": "002.jpg_4", "EncodedPixels": ""},
        ]
        self.csv_path = os.path.join(self.temp_dir, "train.csv")
        pd.DataFrame(records).to_csv(self.csv_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_unet_dataset_mask_shapes(self):
        """Verify SeverstalUNetDataset outputs [4, 224, 224] target masks."""
        dataset = SeverstalUNetDataset(img_dir=self.img_dir, csv_path=self.csv_path)
        self.assertEqual(len(dataset), 2)

        sample0 = dataset[0] if dataset.image_ids[0] == "001.jpg" else dataset[1]
        self.assertEqual(sample0["image"].shape, torch.Size([3, 224, 224]))
        self.assertEqual(sample0["mask"].shape, torch.Size([4, 224, 224]))
        self.assertFalse(sample0["has_defect"])
        self.assertEqual(torch.sum(sample0["mask"]), 0)

        sample1 = dataset[1] if dataset.image_ids[1] == "002.jpg" else dataset[0]
        self.assertTrue(sample1["has_defect"])
        self.assertEqual(sample1["mask"].shape, torch.Size([4, 224, 224]))
        self.assertGreater(torch.sum(sample1["mask"][0]), 0)  # Class 1 defect

    def test_progressive_decoder_forward(self):
        """Verify decoder executes cleanly with standard tensor."""
        decoder = ProgressiveUNetDecoder(in_channels=768, num_classes=4)
        x = torch.randn(1, 768, 16, 16)
        out = decoder(x)
        self.assertEqual(out.shape, torch.Size([1, 4, 224, 224]))


if __name__ == "__main__":
    unittest.main()
