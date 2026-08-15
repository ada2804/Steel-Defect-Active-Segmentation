"""
Unit tests for FAISS IVFPQ memory bank construction and evaluation in src.build_index and src.evaluate.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from src.build_index import build_faiss_ivfpq_index
from src.evaluate import compute_segmentation_metrics, DefectInspector


class TestPipeline(unittest.TestCase):
    """Test suite for FAISS indexing and evaluation pipeline logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test_ivfpq.index")
        self.labels_path = os.path.join(self.temp_dir, "test_labels.npy")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_faiss_ivfpq_build_and_search(self):
        """Verify building FAISS IVFPQ index and nearest neighbor search."""
        num_samples = 400
        dim = 768
        np.random.seed(42)

        # Generate synthetic embeddings and labels (0..4)
        embeddings = np.random.randn(num_samples, dim).astype(np.float32)
        labels = np.random.choice([0, 1, 2, 3, 4], size=num_samples).astype(np.int64)

        # Build index with small nlist for fast test
        index, saved_labels = build_faiss_ivfpq_index(
            embeddings=embeddings,
            labels=labels,
            output_index_path=self.index_path,
            output_labels_path=self.labels_path,
            nlist=10,
            m=16,
            nbits=8,
        )

        self.assertTrue(os.path.exists(self.index_path))
        self.assertTrue(os.path.exists(self.labels_path))
        self.assertEqual(index.ntotal, num_samples)

        # Test search query
        query = embeddings[:5]
        distances, indices = index.search(query, k=3)
        self.assertEqual(indices.shape, (5, 3))
        self.assertEqual(distances.shape, (5, 3))

    def test_metrics_computation(self):
        """Verify Dice and IoU computation on synthetic masks."""
        # 4-channel ground truth and predicted mask
        gt = np.zeros((4, 256, 1600), dtype=np.uint8)
        pred = np.zeros((4, 256, 1600), dtype=np.uint8)

        # Class 1: Perfect match
        gt[0, 10:50, 10:50] = 1
        pred[0, 10:50, 10:50] = 1

        # Class 2: Partial overlap (50% overlap)
        gt[1, 100:200, 100:200] = 1   # Area 10,000
        pred[1, 100:200, 150:250] = 1  # Area 10,000, intersection 5,000

        # Class 3: Both empty
        # Class 4: False positive
        pred[3, 0:10, 0:10] = 1

        metrics = compute_segmentation_metrics(pred, gt)

        # Class 1 dice should be ~1.0
        self.assertAlmostEqual(metrics["dice_class_1"], 1.0, places=2)
        self.assertAlmostEqual(metrics["iou_class_1"], 1.0, places=2)

        # Class 2 dice: 2*5000 / (10000 + 10000) = 0.5
        self.assertAlmostEqual(metrics["dice_class_2"], 0.5, places=2)

        # Class 3 empty should be 1.0
        self.assertAlmostEqual(metrics["dice_class_3"], 1.0, places=2)

        # Class 4 false positive should be ~0.0
        self.assertAlmostEqual(metrics["dice_class_4"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
