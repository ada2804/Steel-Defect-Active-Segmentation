"""
Hybrid Vision Transformer + U-Net Segmentation Model.

Combines a frozen DINOv2 (ViT-B/14) backbone with a lightweight trainable
progressive convolutional decoder for high-precision defect segmentation.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.build_index import extract_patch_embeddings, load_dinov2_model


class ConvBlock(nn.Module):
    """Lightweight Convolutional block with Conv2D, BatchNorm, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ProgressiveUNetDecoder(nn.Module):
    """
    Lightweight Progressive Convolutional Decoder.

    Progressively upsamples folded ViT patch embeddings from [B, 768, 16, 16] to [B, 4, 224, 224].
    """

    def __init__(self, in_channels: int = 768, num_classes: int = 4, target_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.target_size = target_size

        # Channel reduction on 16x16 feature grid: 768 -> 128
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 1: 16x16 -> 32x32
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(64, 64)

        # Block 2: 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(32, 32)

        # Block 3: 64x64 -> 128x128
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(16, 16)

        # Block 4: 128x128 -> 256x256 -> 224x224
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(16, 16)

        # Final Classification Head: Outputs 4 class raw logits (no sigmoid)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x_grid: Tensor of shape [B, 768, 16, 16].

        Returns:
            logits: Tensor of shape [B, num_classes, 224, 224].
        """
        x = self.proj(x_grid)           # [B, 128, 16, 16]
        x = self.conv1(self.up1(x))     # [B, 64, 32, 32]
        x = self.conv2(self.up2(x))     # [B, 32, 64, 64]
        x = self.conv3(self.up3(x))     # [B, 16, 128, 128]
        x = self.up4(x)                 # [B, 16, 256, 256]

        # Interpolate to exact target dimensions (224x224)
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        x = self.conv4(x)
        logits = self.head(x)  # [B, 4, 224, 224]
        return logits


class DinoUNetDecoder(nn.Module):
    """
    Complete Hybrid Segmentation Model:
    Frozen DINOv2 ViT-B/14 Encoder + Trainable Progressive U-Net Decoder.
    """

    def __init__(
        self,
        num_classes: int = 4,
        device: Optional[torch.device] = None,
        load_pretrained_encoder: bool = True,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 1. The Encoder (Frozen Foundation Model)
        if load_pretrained_encoder:
            self.encoder = load_dinov2_model(self.device)
            # Explicitly freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = None

        # 2. The Decoder (Trainable CNN)
        self.decoder = ProgressiveUNetDecoder(in_channels=768, num_classes=num_classes)
        self.decoder.to(self.device)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract [B, 256, 768] patch embeddings from frozen DINOv2."""
        with torch.no_grad():
            patch_tokens = extract_patch_embeddings(self.encoder, images, self.device)
            return patch_tokens.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass: Image [B, 3, 224, 224] -> Defect Logits [B, 4, 224, 224].
        """
        # 1. Extract patch tokens: [B, 256, 768]
        patch_tokens = self.extract_features(images)  # [B, 256, 768]
        batch_size = patch_tokens.shape[0]

        # 2. Fold 256 1D sequence into 16x16 2D spatial feature map: [B, 768, 16, 16]
        # DINOv2 generates 16x16 tokens in row-major order
        x_grid = patch_tokens.view(batch_size, 16, 16, 768).permute(0, 3, 1, 2).contiguous()

        # 3. Decode into pixel-level logits: [B, 4, 224, 224]
        logits = self.decoder(x_grid)
        return logits

    def forward_from_features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Decode directly from pre-extracted [B, 256, 768] patch tokens."""
        batch_size = patch_tokens.shape[0]
        x_grid = patch_tokens.view(batch_size, 16, 16, 768).permute(0, 3, 1, 2).contiguous()
        return self.decoder(x_grid)

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return binary segmentation masks and confidence probabilities.
        """
        self.eval()
        logits = self.forward(images)
        probs = torch.sigmoid(logits)
        binary_masks = (probs >= threshold).float()
        return {
            "logits": logits,
            "probs": probs,
            "binary_masks": binary_masks,
        }
