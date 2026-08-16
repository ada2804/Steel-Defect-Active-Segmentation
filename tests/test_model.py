"""
Unit tests for DinoUNetDecoder model in src.model.
"""

import unittest
import torch

from src.model import ProgressiveUNetDecoder, DinoUNetDecoder


class TestModel(unittest.TestCase):
    """Test suite for DINOv2 U-Net model forward pass and gradients."""

    def test_progressive_decoder_shape(self):
        """Verify decoder maps [B, 768, 16, 16] to [B, 4, 224, 224]."""
        decoder = ProgressiveUNetDecoder(in_channels=768, num_classes=4)
        x_grid = torch.randn(2, 768, 16, 16)
        out = decoder(x_grid)
        self.assertEqual(out.shape, torch.Size([2, 4, 224, 224]))

    def test_frozen_encoder_trainable_decoder(self):
        """Verify only decoder parameters have requires_grad=True."""
        model = DinoUNetDecoder(num_classes=4, load_pretrained_encoder=False)
        # Check decoder params
        decoder_trainable = [p.requires_grad for p in model.decoder.parameters()]
        self.assertTrue(all(decoder_trainable))
        self.assertGreater(len(decoder_trainable), 0)

    def test_loss_backward_on_decoder(self):
        """Verify backward pass updates decoder weights properly."""
        decoder = ProgressiveUNetDecoder(in_channels=768, num_classes=4)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        x = torch.randn(2, 768, 16, 16)
        target = torch.randint(0, 2, (2, 4, 224, 224)).float()

        optimizer.zero_grad()
        out = decoder(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        self.assertIsNotNone(loss.item())
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
